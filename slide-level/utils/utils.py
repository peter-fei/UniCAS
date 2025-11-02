import os
import sys
import json
import pickle
import random
from utils.functions import calculate_loss

import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random
import sys
import torchvision.models as models
from losses.losses import Ralloss
from args import get_args
from torchmetrics import MetricCollection,Recall,Specificity,AUROC,Precision,F1Score,Accuracy,ConfusionMatrix, MeanSquaredError, MeanAbsoluteError
from torch.nn import functional as F
from torchvision.transforms.functional import center_crop
import pandas as pd
args=get_args()


def pre_cls_model(x):
    features = resnet(x)
    #print('sad',features.size())
    return features.squeeze()


softmax = nn.Softmax(dim=1)

device_gpu = torch.device('cuda')
device_cpu = torch.device('cpu')


def get_lossfn(name,use_reduction=False,gamma_neg=4):
    if name == 'CELoss':
        return nn.CrossEntropyLoss(reduction = args.reduction if use_reduction else 'mean')
    elif name == 'FocalLoss':
        return FocalLoss(gamma=2,alpha=0.25,size_average=False)
    elif name == 'BARE':
        # print('BARE LOSS')
        return WeightedCCE(k=1,num_class=2,reduction=args.reduction if use_reduction else 'mean')
    elif name == 'Ral':
        return Ralloss(gamma_neg=gamma_neg)
    elif name == 'MSELoss':
        return nn.MSELoss()


def train_one_epoch(model, optimizer, data_loader, local_rank, epoch, num_tasks,istrain=[True]*args.num_tasks,cont=False):
    model.train()
    # model.module.encoder.eval()
    
    # loss_function = torch.nn.BCELoss()#torch.nn.CrossEntropyLoss()
    loss_functions = ['CELoss'] * args.num_tasks
    accu_loss = torch.zeros(1).to(local_rank)  # 累计损失
    accu_num = torch.zeros(1).to(local_rank)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_num = [0] * num_tasks
    accu_loss = [0] * num_tasks
    errors_nums = [0] * num_tasks
    errors_lists = [ [] for i in range(num_tasks) ]
    task_epoch = [1e-8] * num_tasks
    task_num = [1e-8] * num_tasks
    
    sens_metrics = [ Recall(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    spec_metrics = [ Specificity(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    auc_metrics =  [ AUROC(task='multiclass',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    
    for step, data in enumerate(data_loader):
        images, labels = data
        if torch.isnan(images).any():
            print(labels['name'],'asdsaw')
            continue
        # print(labels)
        sample_num += images.shape[0]
        x = images.cuda(local_rank)
        B = x.shape[0]

        features = x

        preds = model(features)

        # 
        
        task_id = labels['task_id'].cuda(local_rank)
        task_id_mask = torch.zeros(args.num_tasks, task_id.shape[0]).cuda(local_rank)
        # index = torch.Tensor((torch.range(0,args.num_tasks),task_id))
        if args.num_tasks > 1:
            
            task_id_mask[task_id,range(task_id.shape[0])]=1
            task_id_mask = task_id_mask.long()
        else:
            task_id_mask[0,range(task_id.shape[0])]=1
            task_id_mask = task_id_mask.long()

        # print(task_id_mask,'oopop',(list(labels.keys())[:args.num_tasks]))

        code = labels['code']
        loss_total = 0
        # loss_current = []

        a = torch.FloatTensor([0,0,0,0])
        for i,key in enumerate(list(labels.keys())[:args.num_tasks]):
            # print()
            if not istrain[i] or task_id_mask[i].sum().item() == 0:
                continue
            # print(i,key,preds[i])
            pred = preds[i].cuda(local_rank)
            label = labels[key].cuda(local_rank) 
            cancer_grade = labels['cancer_grade'].cuda(local_rank)
            
            # print(cancer_grade.size(),task_id_mask[i].size(),pred.size(),'dsd')
            pred = pred[task_id_mask[i]==1]
            label = label[task_id_mask[i]==1]
            cancer_grade = cancer_grade[task_id_mask[i]==1]
            # labels['code'] = labels['code']
            
            code1 = [code[k] for k in range(len(code)) if task_id_mask[i][k] == 1]
            # print(args.tasks[i],type(labels['code']),len(code1),pred.size(),cancer_grade.size())
            
            loss_function = get_lossfn(loss_functions[i],use_reduction=True)
            labels1 = {'cancer_grade':cancer_grade,'code':code1,'label':label}
            # print(pred,pred.size(),label.size(),label,'lllkkss')
            loss,errors_num,error_name = calculate_loss(loss_function,pred,label,args.tasks[i],args.loss_weights,labels1,cont)
            
            # print(i,loss,code1,accu_loss[i],task_epoch[i],label,code1)
            
            accu_loss[i] += loss.detach().item()
            errors_nums[i] += errors_num
            errors_lists[i].extend(error_name)
             
            pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]
            accu_num[i] += torch.eq(pred_classes, label.cuda(local_rank)).sum().item()
            # print(pred_classes,label,loss)

            task_epoch[i] += 1
            task_num[i] += task_id_mask[i].sum().item()

            sens_metrics[i].update(pred_classes,label)
            spec_metrics[i].update(pred_classes,label)
            auc_metrics[i].update(pred,label)
            # print(pred_classes,label,code1,i)
            
            loss_total +=loss 
        # loss_noisy = collect_noisy_gating_loss(model,weight=0.01)
        # loss_total+=loss_noisy
        loss_total.backward()
        # print(task_num,accu_num)
        # s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,acc_l/ (step + 1),i,acc_n/ sample_num) for i,(acc_l,acc_n) in enumerate(zip(accu_loss,accu_num))])
        s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,accu_loss[i]/ (task_epoch[i]),i,accu_num[i]/ task_num[i]) for i in range(args.num_tasks)])
 
        s_desc = f'[train epoch {epoch}] '+ s 
        
        if cont :
            s_desc += f' error: {errors_nums[0]}'
        data_loader.desc = s_desc
        
        
#        print(loss_total,'wwwwwwwwww')
        if not torch.isfinite(loss_total):
            print('WARNING: non-finite loss, ending training ', loss_total)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()


    sens_res = [sens.compute().item() if istrain[i] else 0 for i,sens in enumerate(sens_metrics)]
    spec_res = [spec.compute().item() if istrain[i] else 0 for i,spec in enumerate(spec_metrics)] 
#    auc_res = [auc.compute().item() if istrain[i] else 0 for i,auc in enumerate(auc_metrics)] 
    auc_res = []
    for i,auc in enumerate(auc_metrics):
# print(len(auc),'iiiiiiiiiiiiiiiiiiiiiiiiii')
        try:
          auc_res.append(auc.compute().item())
        except:
          auc_res.append(0)
    if cont:
        print(f'total_errors_label : {errors_nums[0]} {len(errors_lists[0])} ')
        
    print('Train Sensitive: ',sens_res)
    print('Train Specificity: ',spec_res)
    print('Train AUC: ',auc_res)
    del sens_metrics,spec_metrics,auc_metrics,pred,labels,data,label
    return np.array(accu_loss) / np.array(task_epoch), np.array(accu_num) / np.array(task_num), errors_lists, sens_res, spec_res, auc_res

@torch.no_grad()
def evaluate_all(model, data_loader, local_rank, epoch, num_tasks,name='test',cont=False):
    # loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    # model.train()

    zhenyin = 0
    gjb = 0
    sample_num = 0
    louzhen = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_loss = [0] * num_tasks
    accu_num = [0] * num_tasks
    loss_functions =  ['CELoss'] * args.num_tasks
    errors_nums = [0] * num_tasks
    errors_lists = [ [] for i in range(num_tasks) ]
    task_epoch = [1e-8] * num_tasks
    task_num = [1e-8] * num_tasks
    
    sens_metrics = [ Recall(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    spec_metrics = [ Specificity(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    auc_metrics =  [ AUROC(task='multiclass' if i>1 else 'binary' ,num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    pre_metrics = [ Precision(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    f1_metrics = [ F1Score(task='multiclass',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    acc_metrics = [ Accuracy(task='multiclass',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    
    test_confusion = [ConfusionMatrix(task="binary", num_classes=2).cuda(local_rank) for i in args.classes]
    
    preds_tasks = [ [] for i in range(4)]
    # preds_fungus = []
    # preds_cluecell = []
    preds_code = []

    for step, data in enumerate(data_loader):
        
        images, labels = data
        sample_num += images.shape[0]
        # print(labels)
        x = images.cuda(local_rank)
        
        B = x.shape[0]
        
        features = x
 

        preds = model(features)

        code = labels['code']
        loss_total = 0

        preds_code.extend(code)
        # print(code)
        # exit()


        for i,key in enumerate(list(labels.keys())[:num_tasks]):

            pred = preds[i].cuda(local_rank)
            label = labels[key].cuda(local_rank) 
            cancer_grade = labels['cancer_grade'].cuda(local_rank)
            
            code1 = [code[k] for k in range(len(code))]

            pred_ = torch.softmax(pred, dim=1)[:,1]
            for p in pred_:
                preds_tasks[i].append(round(p.item(),2))

            labels1 = {'cancer_grade':cancer_grade,'code':code1,'label':label}
            loss_function = get_lossfn(loss_functions[i],use_reduction=False)

            loss,errors_num,error_name = calculate_loss(loss_function,pred,label,args.tasks[i],args.loss_weights,labels1,cont)
            accu_loss[i] += loss.detach().item()
            errors_nums[i] += errors_num
            errors_lists[i].extend(error_name)
            
            loss = loss_function(pred,label)

            if args.reduction == 'none':
                loss = loss.mean()
            

            pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]

            accu_num[i] += torch.eq(pred_classes, label).sum().item()
            

            loss_total +=loss
            if args.classes[i] > 2:
                pred_classes = torch.softmax(pred, dim=1)
            sens_metrics[i].update(pred_classes,label)
            spec_metrics[i].update(pred_classes,label)
            auc_metrics[i].update(pred,label)
            pre_metrics[i].update(pred_classes,label)
            f1_metrics[i].update(pred_classes,label)
            acc_metrics[i].update(pred_classes,label)

            test_confusion[i].update(pred_classes, label)

            task_epoch[i] += 1
            task_num[i] += x.size(0)
            
        s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,accu_loss[i]/ (task_epoch[i]),i,accu_num[i]/ task_num[i]) for i in range(args.num_tasks)])
        s_desc = f'[{name} epoch {epoch}] '+ s
        
        if cont :
            s_desc += f' error: {errors_nums[0]}'

        data_loader.desc = s_desc

    sens_res = [sens.compute().item()  for i,sens in enumerate(sens_metrics)  ]
    spec_res = [spec.compute().item() for i,spec in enumerate(spec_metrics) ] 
    auc_res = []
    for i,auc in enumerate(auc_metrics):
        # print(len(auc),'iiiiiiiiiiiiiiiiiiiiiiiiii')
        try:
            auc_res.append(auc.compute().item())
        except:
            auc_res.append(0)
    pre_res = [pre.compute().item() for i,pre in enumerate(pre_metrics) ] 
    f1_res = [f1.compute().item() for i,f1 in enumerate(f1_metrics) ] 
    acc_res = [acc.compute().item() for i,acc in enumerate(acc_metrics) ] 
    confusion_res = [con.compute() for i,con in enumerate(test_confusion)]
    for idx,con in enumerate(confusion_res):
        print(idx,con,'ppopoo') 
        # print(i,'ppppppppppp')
        # print(auc.compute().item())
    # auc_res = [auc.compute().item() for i,auc in enumerate(auc_metrics) ] 
    
    if cont:
        # print(err_res)
        print(f'total_errors_label : {errors_nums[0]} {len(errors_lists[0])} ')
    
    if local_rank == 0:
        print(f'{name} ACC: ',acc_res)
        print(f'{name} AUC: ',auc_res)
        print(f'{name} Senstive: ',sens_res)
        print(f'{name} Specificity: ',spec_res)
        print(f'{name} Precision: ',pre_res)
        print(f'{name} F1_Score: ',f1_res)
    del sens_metrics,spec_metrics,auc_metrics,pred,labels,data,label
    # df = pd.DataFrame({'code':preds_code,'pred_cerv':preds_tasks[0],'pred_candi':preds_tasks[2],'pred_cluecell':preds_tasks[3]})
    # df.to_csv('pred.csv',index=False)
    return np.array(accu_loss) / np.array(task_epoch), np.array(accu_num) / np.array(task_num), errors_lists, sens_res, spec_res, auc_res, f1_res


@torch.no_grad()
def evaluate(model, data_loader, local_rank, epoch, num_tasks,name='valid',cont=False,text_features=None):
    # loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    # model.train()

    zhenyin = 0
    gjb = 0
    sample_num = 0
    louzhen = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_loss = [0] * num_tasks
    accu_num = [0] * num_tasks
    loss_functions =  ['CELoss'] * args.num_tasks
    errors_nums = [0] * num_tasks
    errors_lists = [ [] for i in range(num_tasks) ]
    task_epoch = [1e-8] * num_tasks
    task_num = [1e-8] * num_tasks
    
    sens_metrics = [ Recall(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    spec_metrics = [ Specificity(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    auc_metrics =  [ AUROC(task='multiclass' if i>1 else 'binary' ,num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    pre_metrics = [ Precision(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    f1_metrics = [ F1Score(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    acc_metrics = [ Accuracy(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.classes ]
    
    test_confusion = [ConfusionMatrix(task="binary", num_classes=2).cuda(local_rank) for i in args.classes]

    res_preds = [[] for i in range(num_tasks)]
    res_labels = [ [] for i in range(num_tasks)]

    names = []
    label_total = []
    pred_res = []
    task_ids = []
    for step, data in enumerate(data_loader):
        
        images, labels = data
        sample_num += images.shape[0]
        # print(labels)
        x = images.cuda(local_rank)
        
        B = x.shape[0]

        features = x

        if text_features is not None:
            preds = model(features,text_features)
            
        else:
            preds = model(features)

        code = labels['name']
        loss_total = 0

        task_id = labels['task_id'].cuda(local_rank)
        task_id_mask = torch.zeros(args.num_tasks, task_id.shape[0]).cuda(local_rank)
        if args.num_tasks > 1:
            task_id_mask[task_id,range(task_id.shape[0])]=1
            task_id_mask = task_id_mask.long()
        else:
            task_id_mask[0,range(task_id.shape[0])]=1
            task_id_mask = task_id_mask.long()

        if args.save_pred:
            names.extend(code)
            

        for i,key in enumerate(list(labels.keys())[:num_tasks]):
            if args.save_pred:
                res_preds[i].extend(preds[i].cpu().detach().tolist())
                res_labels[i].extend(labels[key].cpu().detach().tolist())
                task_ids.extend([i]*(task_id_mask[i]==1).sum())

            if task_id_mask[i].sum().item() == 0:
                continue

            pred = preds[i].cuda(local_rank)
            label = labels[key].cuda(local_rank) 
            cancer_grade = labels['cancer_grade'].cuda(local_rank)

            label = label[task_id_mask[i]==1]
            cancer_grade = cancer_grade[task_id_mask[i]==1]


            if text_features is not None:
                text_feature = preds[f'text_{i}'].cuda(local_rank)
                pred,text_feature = get_logits(pred,text_feature)

            pred = pred[task_id_mask[i]==1]
            
            code1 = [code[k] for k in range(len(code)) if task_id_mask[i][k] == 1]
    
            labels1 = {'cancer_grade':cancer_grade,'code':code1,'label':label}
            loss_function = get_lossfn(loss_functions[i],use_reduction=False)

            loss,errors_num,error_name = calculate_loss(loss_function,pred,label,args.tasks[i],args.loss_weights,labels1,cont)
            accu_loss[i] += loss.detach().item()
            errors_nums[i] += errors_num
            errors_lists[i].extend(error_name)
            
            loss = loss_function(pred,label)

            if pred.shape[-1] == 1:
                pred_classes = (pred.sigmoid()>0.5).float().squeeze()
            else:
                pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]

            accu_num[i] += torch.eq(pred_classes, label).sum().item()

            loss_total +=loss
            if args.classes[i] > 2:
                pred_classes = torch.softmax(pred, dim=1)
            sens_metrics[i].update(pred_classes,label)
            spec_metrics[i].update(pred_classes,label)
            auc_metrics[i].update(pred,label)
            pre_metrics[i].update(pred_classes,label)
            f1_metrics[i].update(pred_classes,label)
            acc_metrics[i].update(pred_classes,label)

            test_confusion[i].update(pred_classes, label)
            # print(pred_classes,label,code1,i)

            task_epoch[i] += 1
            task_num[i] += task_id_mask[i].sum().item()
            
        
        s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,accu_loss[i]/ (task_epoch[i]),i,accu_num[i]/ task_num[i]) for i in range(args.num_tasks)])
        s_desc = f'[{name} epoch {epoch}] '+ s
        
        if cont :
            s_desc += f' error: {errors_nums[0]}'

        data_loader.desc = s_desc

    sens_res = [sens.compute().item()  for i,sens in enumerate(sens_metrics)  ]
    spec_res = [spec.compute().item() for i,spec in enumerate(spec_metrics) ] 
    auc_res = []
    for i,auc in enumerate(auc_metrics):
        # print(len(auc),'iiiiiiiiiiiiiiiiiiiiiiiiii')
        try:
            auc_res.append(auc.compute().item())
        except:
            auc_res.append(0)
    pre_res = [pre.compute().item() for i,pre in enumerate(pre_metrics) ] 
    f1_res = [f1.compute().item() for i,f1 in enumerate(f1_metrics) ] 
    acc_res = [acc.compute().item() for i,acc in enumerate(acc_metrics) ] 
    confusion_res = [con.compute() for i,con in enumerate(test_confusion)]
    for idx,con in enumerate(confusion_res):
        print(idx,con,'ppopoo') 
        # print(i,'ppppppppppp')
        # print(auc.compute().item())
    # auc_res = [auc.compute().item() for i,auc in enumerate(auc_metrics) ] 
    
    if cont:
        # print(err_res)
        print(f'total_errors_label : {errors_nums[0]} {len(errors_lists[0])} ')
    
    if local_rank == 0:
        print(f'{name} ACC: ',acc_res)
        print(f'{name} AUC: ',auc_res)
        print(f'{name} Senstive: ',sens_res)
        print(f'{name} Specificity: ',spec_res)
        print(f'{name} Precision: ',pre_res)
        print(f'{name} F1_Score: ',f1_res)
    del sens_metrics,spec_metrics,auc_metrics,pred,labels,data,label

    
    if args.save_pred:
        # for i in range(4):
        #     if len(res_preds[i]) > 0:
        #         res_preds[i] = torch.cat(res_preds[i],dim=0)
        #         res_labels[i] = torch.cat(res_labels[i],dim=0)
        #         print(i,res_preds[i].shape,res_labels[i].shape,'sad')

        res_dict = {}
        res_dict['name'] = names
        res_dict['task_id'] = task_ids
        for i in range(num_tasks):
            res_dict[args.tasks[i]] = res_preds[i]
            res_dict[f'{args.tasks[i]}_label'] = res_labels[i]
        # res_dict = {f'{args.tasks[i]}':res_preds,f'{args.tasks[i]}_label':res_labels[i] for i in range(num_tasks)}
        df = pd.DataFrame(res_dict)
        # df.to_csv('/public_bme2/bme-wangqian2/jianght/ckpts/FOLD/Age/uni_transmil_age_(46,55)/model_transmil_17_cls.csv',index=False)
        df.to_csv(f'/public_bme2/bme-wangqian2/jianght/ckpts/Pred_res/{args.model}/{args.encoder}_{args.tasks}.csv',index=False)
        # import pickle
        # os.makedirs('res_tsne',exist_ok=True)
        # encoder = args.encoder
        # # with open(f'res_tsne/pred_{encoder}_waibu.pkl','wb') as f:
        # #     pickle.dump(res_preds,f)
        # # with open(f'res_tsne/label_{encoder}_waibu.pkl','wb') as f:
        # #     pickle.dump(res_labels,f)
        # with open(f'res_tsne/pred_{encoder}_gen_train.pkl','wb') as f:
        #     pickle.dump(res_preds,f)
        # with open(f'res_tsne/label_{encoder}_gen_train.pkl','wb') as f:
        #     pickle.dump(res_labels,f)
        # exit()
    return np.array(accu_loss) / np.array(task_epoch), np.array(accu_num) / np.array(task_num), errors_lists, sens_res, spec_res, auc_res, f1_res



# @torch.no_grad()
@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.float16)
def extract_feature(model, data_loader, local_rank):
    # loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    # model.train()

    for step, data in enumerate(tqdm(data_loader)):
        
        images, labels = data

        x = images.cuda(local_rank)
        if len(x.shape) == 3:
            if not x.isnan().any():
                # print(x.size())
                continue
            else:
                raise 'oopopo'
        B = x.shape[0]
        

        # if len(x.shape) == 5:
        #     x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = x

        code = labels['name']
        for i in range(len(code)):
            folder_path = get_folder_path(code[i])
            # save_path = folder_path.replace("/public_bme/data/jianght/datas/Pathology",f'/public_bme/data/jianght/datas/Pathology_{args.encoder}_patch')
            # save_path = folder_path.replace('Pathology',f"Pathology_{args.encoder}_stainNA_2")
            # save_path = folder_path.replace("/public_bme/data/jianght/datas/Pathology",f'/public_bme2/bme-wangqian2/jianght/Data_Pathology/Pathology_{args.encoder}_Slide_Slicer003_1')
            # save_path = folder_path.replace("/public_bme/data/jianght/datas/Pathology",f'/public_bme2/bme-wangqian2/jianght/Data_Pathology/Pathology_{args.encoder}_stainNorm_1')
            save_path = folder_path.replace("/public_bme/data/jianght/datas/Pathology",f'/public_bme2/bme-wangqian2/jianght/Data_Pathology/Pathology_{args.encoder}_Slide_1')
            # save_path = folder_path.replace("/public_bme/data/jianght/datas/Pathology",f'/public_bme2/bme-wangqian2/jianght/Data_Pathology/Pathology_{args.encoder}_patch')
            # print(save_path)
            # save_path = folder_path.replace(IMAGE_FOLDER,f'/public_bme2/bme-wangqian2/jianght/Data_Pathology/Pathology_{self.feature_name}_stainNA_1')
            # print(save_path,folder_path,f"Pathology_{args.encoder}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                if features[i].sum()==0:
                    continue
            if os.path.isfile(save_path+'/images.pt') and os.path.getsize(save_path+'/images.pt') > 0:
                a = torch.load(save_path+'/images.pt')
                if a.size(0) ==800 and not a.isnan().any():
                    continue
                else:
                    print(a.size())

            feature = features[i]
            # feature = F.interpolate(feature, size=(224, 224), mode='bilinear', align_corners=False)
            feature = center_crop(feature,(224,224))
            # if args.encoder == 'conch':
            #     preds = model.module.encode_image(feature, proj_contrast=False, normalize=False)
            # else:
            # preds = model(feature).last_hidden_state[:, 0, :].squeeze()
            preds = model(feature).squeeze()
            # print(feature.size(),preds.size(),preds.isnan().any(),preds.sum(),save_path,code[i])
            # exit()
            # print(preds.size())
            # exit()
            if len(preds.shape) == 3:
                preds = preds.mean(1)
            if preds.isnan().any():
                raise f'{save_path} error get NULL'
            # save_path = save_path.replace('/public_bme/data/jianght/datas','/public/home/jianght2023/datas')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # print(feature.size(),preds.size(),preds.isnan().any(),preds.sum(),preds.max().item(),preds.min().item(),save_path)
            # exit()
            
            torch.save(preds.detach().cpu(),save_path+'/images.pt')



@torch.no_grad()
def get_text_feature(clip,text):
    text_feature = clip.encode_text(text)
    return text_feature

def get_folder_path(name,floder="/public_bme/data/jianght/datas/Pathology"):

    folder_path = None
    if os.path.isdir(os.path.join(floder, name, 'torch')):
        folder_path = os.path.join(floder, name, 'torch')
    elif os.path.isdir(os.path.join(floder,'yangxing','Torch',name,'torch')):
        folder_path = os.path.join(floder,'yangxing','Torch',name,'torch')
    elif os.path.isdir(os.path.join(floder,'yinxing','Torch',name,'torch')):
        folder_path = os.path.join(floder,'yinxing','Torch',name,'torch')
    elif os.path.isdir(os.path.join(floder, name)):
        folder_path = os.path.join(floder, name)
    elif os.path.isdir(os.path.join(floder,'Degree',name,'torch')):
        folder_path = os.path.join(floder,'Degree',name,'torch')
    elif os.path.isdir(os.path.join(floder,'Degree', name)):
        folder_path = os.path.join(floder,'Degree', name)
    return folder_path

def collect_noisy_gating_loss(model, weight):
    loss = 0
    for module in model.modules():
        if (isinstance(module, NoisyGate) or isinstance(module, NoisyGate_VMoE)) and module.has_loss:
            # print(module)
            loss += module.get_loss()
    return loss * weight