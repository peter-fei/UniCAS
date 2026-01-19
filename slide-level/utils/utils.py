import os
import sys
from utils.functions import calculate_loss
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from losses.losses import Ralloss, FocalLoss
from torchmetrics import Recall, Specificity, AUROC, Precision, F1Score, Accuracy, ConfusionMatrix
from torchvision.transforms.functional import center_crop
import pandas as pd


def get_lossfn(name, args, use_reduction=False, gamma_neg=4):
    if name == 'CELoss':
        return nn.CrossEntropyLoss(reduction=args.reduction if use_reduction else 'mean')
    elif name == 'FocalLoss':
        return FocalLoss(gamma=2, alpha=0.25, size_average=False)
    elif name == 'BARE':
        raise ValueError("Loss 'BARE' is not supported")
    elif name == 'Ral':
        return Ralloss(gamma_neg=gamma_neg)
    elif name == 'MSELoss':
        return nn.MSELoss()


def initialize_metrics(args, local_rank, include_extra=False):
    """Initialize all metrics for training/evaluation."""
    sens_metrics = [Recall(task='multiclass' if i>2 else 'binary', num_classes=i, average='macro').cuda(local_rank) for i in args.classes]
    spec_metrics = [Specificity(task='multiclass' if i>2 else 'binary', num_classes=i, average='macro').cuda(local_rank) for i in args.classes]
    auc_metrics = [AUROC(task='multiclass', num_classes=i, average='macro').cuda(local_rank) for i in args.classes]
    
    if include_extra:
        pre_metrics = [Precision(task='multiclass' if i>2 else 'binary', num_classes=i, average='macro').cuda(local_rank) for i in args.classes]
        f1_metrics = [F1Score(task='multiclass' if i>2 else 'binary', num_classes=i, average='macro').cuda(local_rank) for i in args.classes]
        acc_metrics = [Accuracy(task='multiclass' if i>2 else 'binary', num_classes=i, average='macro').cuda(local_rank) for i in args.classes]
        confusion_matrices = [ConfusionMatrix(task=('multiclass' if i>2 else 'binary'), num_classes=i).cuda(local_rank) for i in args.classes]
        return sens_metrics, spec_metrics, auc_metrics, pre_metrics, f1_metrics, acc_metrics, confusion_matrices
    
    return sens_metrics, spec_metrics, auc_metrics


def create_task_mask(task_id, num_tasks, batch_size, device):
    """Create task ID mask for multi-task learning."""
    task_id_mask = torch.zeros(num_tasks, batch_size).to(device)
    if num_tasks > 1:
        task_id_mask[task_id, range(batch_size)] = 1
    else:
        task_id_mask[0, range(batch_size)] = 1
    return task_id_mask.long()

def compute_metric_results(metrics_list, istrain=[True]*3):
    metrics_res= []
    for metrics in metrics_list:
        metrics_res.append([metrics[i].compute().item() if istrain[i] else 0 for i in range(len(metrics))])
    return metrics_res


def save_predictions(args, names, task_ids, res_preds, res_labels, num_tasks):
    """Save predictions to CSV file."""
    res_dict = {
        'name': names,
        'task_id': task_ids
    }
    for i in range(num_tasks):
        res_dict[args.tasks[i]] = res_preds[i]
        res_dict[f'{args.tasks[i]}_label'] = res_labels[i]
    
    df = pd.DataFrame(res_dict)
    save_path = f'./Pred_res/{args.model}/{args.encoder}_{args.tasks}.csv'
    df.to_csv(save_path, index=False)


def train_one_epoch(model, optimizer, data_loader, local_rank, epoch, args, istrain=[True]*3, cont=False):
    model.train()
    num_tasks = args.num_tasks
    loss_functions = ['CELoss'] * num_tasks
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_num = [0] * num_tasks
    accu_loss = [0] * num_tasks
    errors_nums = [0] * num_tasks
    errors_lists = [[] for _ in range(num_tasks)]
    task_epoch = [1e-8] * num_tasks
    task_num = [1e-8] * num_tasks

    sens_metrics, spec_metrics, auc_metrics = initialize_metrics(args, local_rank)

    for step, data in enumerate(data_loader):
        images, labels = data
        if torch.isnan(images).any():
            continue
        
        features = images.cuda(local_rank)
        preds = model(features)

        code = labels['code']
        task_id = labels['task_id'].cuda(local_rank)
        task_id_mask = create_task_mask(task_id, num_tasks, task_id.shape[0], features.device)

        loss_total = 0
        for i, key in enumerate(list(labels.keys())[:num_tasks]):
            if not istrain[i] or task_id_mask[i].sum().item() == 0:
                continue
            
            pred = preds[i][task_id_mask[i]==1]
            label = labels[key].cuda(local_rank)[task_id_mask[i]==1]
            cancer_grade = labels['cancer_grade'].cuda(local_rank)[task_id_mask[i]==1]

            loss_function = get_lossfn(loss_functions[i], args, use_reduction=True)
            labels1 = {'cancer_grade': cancer_grade, 'code': code, 'label': label}
            loss, errors_num, error_name = calculate_loss(loss_function, pred, label, args.tasks[i], args.loss_weights, labels1, cont)

            accu_loss[i] += loss.detach().item()
            errors_nums[i] += errors_num
            errors_lists[i].extend(error_name)

            pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]
            accu_num[i] += torch.eq(pred_classes, label).sum().item()

            task_epoch[i] += 1
            task_num[i] += task_id_mask[i].sum().item()

            sens_metrics[i].update(pred_classes, label)
            spec_metrics[i].update(pred_classes, label)
            auc_metrics[i].update(pred, label)

            loss_total += loss
        
        loss_total.backward()
        
        s = ''.join([f' loss_{i}: {accu_loss[i] / task_epoch[i]:.3f}, acc_{i}: {accu_num[i] / task_num[i]:.3f}' 
                     for i in range(num_tasks)])
        s_desc = f'[train epoch {epoch}] {s}'
        if cont:
            s_desc += f' error: {errors_nums[0]}'
        data_loader.desc = s_desc

        if not torch.isfinite(loss_total):
            print('WARNING: non-finite loss, ending training ', loss_total)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    sens_res, spec_res, auc_res = compute_metric_results([sens_metrics, spec_metrics, auc_metrics], istrain)
    
    if cont:
        print(f'total_errors_label: {errors_nums[0]} {len(errors_lists[0])}')

    print('Train Sensitive:', sens_res)
    print('Train Specificity:', spec_res)
    print('Train AUC:', auc_res)
    
    del sens_metrics, spec_metrics, auc_metrics, preds, labels, data
    return (np.array(accu_loss) / np.array(task_epoch), 
            np.array(accu_num) / np.array(task_num), 
            errors_lists, sens_res, spec_res, auc_res)


@torch.no_grad()
def evaluate_all(model, data_loader, local_rank, epoch, args, name='test', cont=False):
    model.eval()
    num_tasks = args.num_tasks
    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_loss = [0] * num_tasks
    accu_num = [0] * num_tasks
    loss_functions = ['CELoss'] * num_tasks
    errors_nums = [0] * num_tasks
    errors_lists = [[] for _ in range(num_tasks)]
    task_epoch = [1e-8] * num_tasks
    task_num = [1e-8] * num_tasks

    sens_metrics, spec_metrics, auc_metrics, pre_metrics, f1_metrics, acc_metrics, confusion_matrices = \
        initialize_metrics(args, local_rank, include_extra=True)
    
    res_preds = [[] for _ in range(num_tasks)]
    res_labels = [[] for _ in range(num_tasks)]
    names = []
    task_ids = []

    for step, data in enumerate(data_loader):
        images, labels = data
        features = images.cuda(local_rank)
        preds = model(features)

        code = labels['code']
        task_id = labels['task_id'].cuda(local_rank)        
        if args.save_pred:
            names.extend(code)

        for i, key in enumerate(list(labels.keys())[:num_tasks]):
            pred = preds[i]
            label = labels[key].cuda(local_rank)
            cancer_grade = labels['cancer_grade'].cuda(local_rank)
            
            if args.save_pred:
                res_preds[i].extend(preds[i].cpu().detach().tolist())
                res_labels[i].extend(label.cpu().detach().tolist())
                task_ids.extend([i] * label.size(0))
            
            code1 = code
            labels1 = {'cancer_grade': cancer_grade, 'code': code1, 'label': label}
            
            loss_function = get_lossfn(loss_functions[i], args, use_reduction=False)
            loss, errors_num, error_name = calculate_loss(loss_function, pred, label, args.tasks[i], 
                                                         args.loss_weights, labels1, cont)
            accu_loss[i] += loss.detach().item()
            errors_nums[i] += errors_num
            errors_lists[i].extend(error_name)

            loss = loss_function(pred, label)
            if args.reduction == 'none':
                loss = loss.mean()
            
            if pred.shape[-1] == 1:
                pred_classes = (pred.sigmoid()>0.5).float().squeeze()
            else:
                pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]
            
            accu_num[i] += torch.eq(pred_classes, label).sum().item()
            
            sens_metrics[i].update(pred_classes, label)
            spec_metrics[i].update(pred_classes, label)
            auc_metrics[i].update(pred, label)
            pre_metrics[i].update(pred_classes, label)
            f1_metrics[i].update(pred_classes, label)
            acc_metrics[i].update(pred_classes, label)
            confusion_matrices[i].update(pred_classes, label)
            
            task_epoch[i] += 1
            task_num[i] += label.size(0)
              
        s = ''.join([f' loss_{i}: {accu_loss[i] / task_epoch[i]:.3f}, acc_{i}: {accu_num[i] / task_num[i]:.3f}' 
                     for i in range(num_tasks)])
        s_desc = f'[{name} epoch {epoch}] {s}'
        if cont:
            s_desc += f' error: {errors_nums[0]}'
        data_loader.desc = s_desc

    acc_res, auc_res, f1_res, sens_res, spec_res, pre_res = compute_metric_results([acc_metrics, auc_metrics, f1_metrics, sens_metrics, spec_metrics, pre_metrics])
    confusion = [confusion_matrices[i].compute() for i in range(len(confusion_matrices))]
    if cont:
        print(f'total_errors_label: {errors_nums[0]} {len(errors_lists[0])}')

    if local_rank == 0:
        print(f'{name} ACC:', acc_res)
        print(f'{name} AUC:', auc_res)
        print(f'{name} Senstive:', sens_res)
        print(f'{name} Specificity:', spec_res)
        print(f'{name} Precision:', pre_res)
        print(f'{name} F1_Score:', f1_res)
        for i in range(num_tasks):
            print(f'{name} {args.tasks[i]} Confusion Matrix: \n', confusion[i])
    
    del sens_metrics, spec_metrics, auc_metrics, preds, labels, data

    if args.save_pred:
        save_predictions(args, names, task_ids, res_preds, res_labels, num_tasks)
    
    return (np.array(accu_loss) / np.array(task_epoch), 
            np.array(accu_num) / np.array(task_num), 
            errors_lists, sens_res, spec_res, auc_res, f1_res)


@torch.no_grad()
def evaluate(model, data_loader, local_rank, epoch, args, name='valid', cont=False, text_features=None):
    model.eval()
    num_tasks = args.num_tasks
    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_loss = [0] * num_tasks
    accu_num = [0] * num_tasks
    loss_functions = ['CELoss'] * num_tasks
    errors_nums = [0] * num_tasks
    errors_lists = [[] for _ in range(num_tasks)]
    task_epoch = [1e-8] * num_tasks
    task_num = [1e-8] * num_tasks

    sens_metrics, spec_metrics, auc_metrics, pre_metrics, f1_metrics, acc_metrics, confusion_matrices = \
        initialize_metrics(args, local_rank, include_extra=True)
    
    res_preds = [[] for _ in range(num_tasks)]
    res_labels = [[] for _ in range(num_tasks)]
    names = []
    task_ids = []

    for step, data in enumerate(data_loader):
        images, labels = data
        features = images.cuda(local_rank)

        if text_features is not None:
            preds = model(features, text_features)
        else:
            preds = model(features)

        code = labels['name']
        task_id = labels['task_id'].cuda(local_rank)
        task_id_mask = create_task_mask(task_id, num_tasks, task_id.shape[0], features.device)

        if args.save_pred:
            names.extend(code)

        for i, key in enumerate(list(labels.keys())[:num_tasks]):
            if args.save_pred:
                res_preds[i].extend(preds[i].cpu().detach().tolist())
                res_labels[i].extend(labels[key].cpu().detach().tolist())
                task_ids.extend([i] * (task_id_mask[i] == 1).sum())

            if task_id_mask[i].sum().item() == 0:
                continue

            pred = preds[i][task_id_mask[i]==1]
            label = labels[key].cuda(local_rank)[task_id_mask[i]==1]
            cancer_grade = labels['cancer_grade'].cuda(local_rank)[task_id_mask[i]==1]

            code1 = [code[k] for k in range(len(code)) if task_id_mask[i][k] == 1]
            labels1 = {'cancer_grade': cancer_grade, 'code': code1, 'label': label}
            
            loss_function = get_lossfn(loss_functions[i], args, use_reduction=False)
            loss, errors_num, error_name = calculate_loss(loss_function, pred, label, args.tasks[i], 
                                                         args.loss_weights, labels1, cont)
            accu_loss[i] += loss.detach().item()
            errors_nums[i] += errors_num
            errors_lists[i].extend(error_name)

            if pred.shape[-1] == 1:
                pred_classes = (pred.sigmoid()>0.5).float().squeeze()
            else:
                pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]

            accu_num[i] += torch.eq(pred_classes, label).sum().item()

            if args.classes[i] > 2:
                pred_classes = torch.softmax(pred, dim=1)
            
            sens_metrics[i].update(pred_classes, label)
            spec_metrics[i].update(pred_classes, label)
            auc_metrics[i].update(pred, label)
            pre_metrics[i].update(pred_classes, label)
            f1_metrics[i].update(pred_classes, label)
            acc_metrics[i].update(pred_classes, label)
            confusion_matrices[i].update(pred_classes, label)
            task_epoch[i] += 1
            task_num[i] += task_id_mask[i].sum().item()

        s = ''.join([f' loss_{i}: {accu_loss[i] / task_epoch[i]:.3f}, acc_{i}: {accu_num[i] / task_num[i]:.3f}' 
                     for i in range(num_tasks)])
        s_desc = f'[{name} epoch {epoch}] {s}'
        if cont:
            s_desc += f' error: {errors_nums[0]}'
        data_loader.desc = s_desc

    acc_res, auc_res, f1_res, sens_res, spec_res, pre_res = compute_metric_results([acc_metrics, auc_metrics, f1_metrics, sens_metrics, spec_metrics, pre_metrics])
    confusion = [confusion_matrices[i].compute() for i in range(len(confusion_matrices))]
    pre_res = [pre.compute().item() for pre in pre_metrics]
    f1_res = [f1.compute().item() for f1 in f1_metrics]
    acc_res = [acc.compute().item() for acc in acc_metrics]

    if cont:
        print(f'total_errors_label: {errors_nums[0]} {len(errors_lists[0])}')

    if local_rank == 0:
        print(f'{name} ACC:', acc_res)
        print(f'{name} AUC:', auc_res)
        print(f'{name} Senstive:', sens_res)
        print(f'{name} Specificity:', spec_res)
        print(f'{name} Precision:', pre_res)
        print(f'{name} F1_Score:', f1_res)
        for i in range(num_tasks):
            print(f'{name} {args.tasks[i]} Confusion Matrix: \n', confusion[i])
    
    del sens_metrics, spec_metrics, auc_metrics, preds, labels, data

    if args.save_pred:
        save_predictions(args, names, task_ids, res_preds, res_labels, num_tasks)
    
    return (np.array(accu_loss) / np.array(task_epoch), acc_res, sens_res, spec_res, auc_res, f1_res, errors_lists)


@torch.inference_mode()
def extract_feature(model, data_loader, device):
    model.eval()
    for step, data in enumerate(tqdm(data_loader)):
        images, save_paths = data
        images = images.to(device)
        
        for i in range(len(save_paths)):
            save_path = save_paths[i]
            image = images[i]
            
            # Skip empty images
            if image.sum() == 0:
                continue
            
            # Check if feature already exists and is valid
            feature_file = os.path.join(save_path, 'images.pt')
            if os.path.isfile(feature_file) and os.path.getsize(feature_file) > 0:
                try:
                    a = torch.load(feature_file)
                    if a.size(0) >= 800 and not a.isnan().any(): #The min num of patches is 800
                        continue
                except:
                    pass
            # Extract features
            feature = center_crop(image, (224, 224))
            preds = model(feature).squeeze()
            
            if len(preds.shape) == 3:
                preds = preds[:, 0]
            if preds.isnan().any():
                raise ValueError(f'{save_path} error get NULL')

            # Save features
            os.makedirs(save_path, exist_ok=True)
            torch.save(preds.detach().cpu(), feature_file)
