import warnings
import os
import math
from utils.functions import get_optimizer

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DistributedSampler
from args import get_args
from my_dataset import MultiDataSet
# from models.vit_res_model import *
from models.slide_multitask import Slide_Multitask
from utils.utils import  train_one_epoch, evaluate, evaluate_all

import pandas as pd
import random
import numpy as np
import warnings


warnings.filterwarnings('ignore')

def seed_reproducer(seed=2022):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

def main(args):
    print(args.local_rank)
    print('-'*100)
    seed_reproducer(9)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = MultiDataSet(data=pd.read_csv(args.train_csv),
                              img_batch=args.img_batch,
                              tasks=args.tasks,
                              task_id=args.task_id,
                              encoder = args.encoder,
                                 )

    val_dataset = MultiDataSet(data=pd.read_csv(args.valid_csv),
                                img_batch=args.img_batch,
                                tasks=args.tasks,
                                task_id=args.task_id,
                                encoder = args.encoder,
                                )
                                
    test_dataset = MultiDataSet(data=pd.read_csv(args.test_csv),
                                img_batch=args.img_batch,
                                tasks=args.tasks,
                                encoder = args.encoder,
                                )


    print('len_dataset: ',len(train_dataset),len(val_dataset),len(test_dataset))
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               num_workers=nw,
                                               sampler = train_sampler
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                            num_workers=nw,
                                             )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,)


    model = Slide_Multitask(args=args).to(device)

    

    if args.weights != "":
        print(f'loading ckpt from {args.weights} --------------------',model.load_state_dict(torch.load(args.weights), strict=False))

   
    optimizer = get_optimizer(args,model)


    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    istrain_list = [True  if args.lr_head[i] > 1e-8 else False for i in range(args.num_tasks)]
    print(istrain_list)

    model = nn.parallel.DistributedDataParallel(model.cuda(args.local_rank),device_ids=[args.local_rank], find_unused_parameters=True,broadcast_buffers=False)

    if args.cont:
        with open(args.logdir,'w') as f:
            f.write('')
    train_error_dict = {}
    val_error_dict = {}
    test_error_dict = {}
    train_error_list = []
    test_error_list = []
    if not args.eval_only:
        tb_writer = SummaryWriter(args.where)
        print(args.where,tb_writer,'pooo',args.cont, args.local_rank)
    train_loss = None
    

    num_per_cls_list = train_dataset.num_per_cls_list


    if args.eval_only:
        val_loss, val_accs, val_error_list, val_senss, val_specs,val_aucs,val_f1s = evaluate(model=model,
                            data_loader=val_loader,
                            local_rank=args.local_rank,
                            epoch=epoch,
                            num_tasks=args.num_tasks,
                            name='val',
                            cont=True,
                        )
        epoch = 0
        test_loss, test_accs, test_error_list, test_senss, test_specs,test_aucs,test_f1s = evaluate_all(model=model,
                        data_loader=test_loader,
                        local_rank=args.local_rank,
                        epoch=epoch,
                        num_tasks=args.num_tasks,
                        name='test',
                        cont=True,
                    )
        exit()

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)

        if not args.eval_only:
            train_loss, train_accs, train_error_list, train_senss, train_specs,train_aucs = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    local_rank=args.local_rank,
                                    epoch=epoch,
                                    num_tasks=args.num_tasks,
                                    istrain = istrain_list,
                                    cont=args.cont,
                                    )
    # #   
            scheduler.step()
            if args.local_rank ==0:
                torch.save(model.module.state_dict(), args.where+"/model-{}.pth".format(epoch))  
                
        if epoch % 1 == 0:

            val_loss, val_accs, val_error_list, val_senss, val_specs,val_aucs,val_f1s = evaluate(model=model,
                            data_loader=val_loader,
                            local_rank=args.local_rank,
                            epoch=epoch,
                            num_tasks=args.num_tasks,
                            name='val',
                            cont=True,
                        )
        
            if args.cont and args.local_rank == 0:
                index = args.tasks.index('label') if 'label' in args.tasks else -1
                print(index,args.local_rank,'label')
                train_error_list_label = train_error_list[index]
                val_error_list = val_error_list[index]
                # test_error_list = test_error_list[index]
                with open(args.logdir,'a') as f:
                    if len(train_error_list_label) > 0:
                        for i in train_error_list_label:
                            train_error_dict[i] = train_error_dict.get(i,0)+1
                        train_cont_lines = [str(k).strip() + ': ' + str(v) + '\n' for k,v in train_error_dict.items()]
                        f.write('**'*20 + f'train epoch {epoch}' + '**'*20 +'\n')
                        f.writelines(train_cont_lines)
                        
                    if len(val_error_list) > 0:
                        for i in val_error_list:
                            val_error_dict[i] = val_error_dict.get(i,0)+1
                        val_cont_lines = [str(k).strip() + ': ' + str(v) + '\n' for k,v in val_error_dict.items()]    
                        f.write(f'------------------------------eval epoch {epoch}-----------------------------------\n')
                        f.writelines(val_cont_lines)

            
                tb_writer.add_scalar(f'train_error_highpos',len(train_error_list),epoch)
                tb_writer.add_scalar(f'val_error_highpos',len(val_error_list),epoch)
            

                for i in range(args.num_tasks):
                    tb_writer.add_scalar(f'val_loss_{i}',val_loss[i],epoch)
                    tb_writer.add_scalar(f'val_acc_{i}',val_accs[i],epoch)
                    tb_writer.add_scalar(f'val_sen_{i}',val_senss[i],epoch)
                    tb_writer.add_scalar(f'val_spec_{i}',val_specs[i],epoch)
                    tb_writer.add_scalar(f'val_auc_{i}',val_aucs[i],epoch)


        for i in range(args.num_tasks):
                tb_writer.add_scalar(f'train_loss_{i}',train_loss[i],epoch)
                tb_writer.add_scalar(f'train_acc_{i}',train_accs[i],epoch)
                tb_writer.add_scalar(f'train_sen_{i}',train_senss[i],epoch)
                tb_writer.add_scalar(f'train_spec_{i}',train_specs[i],epoch)             
                tb_writer.add_scalar(f'train_auc_{i}',train_aucs[i],epoch)

        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

        torch.cuda.empty_cache()

        del train_loss, train_accs, train_error_list, train_senss, train_specs,train_aucs
        

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    opt = get_args()
    opt.local_rank = int(os.environ["LOCAL_RANK"])
    print(opt)
    if opt.local_rank == 0 :
        if not os.path.exists(opt.where) :
            print(f'making dir {opt.where}')
            os.makedirs(opt.where)

    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(opt.local_rank)

    main(opt)
