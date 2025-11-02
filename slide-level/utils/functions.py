import numpy as np
import torch
from torch import optim
import os

def calculate_loss(lossfn,pred,label,task_name,loss_weight,labels,cont,text_feature=None,reduction='none'):

    if label.dtype != torch.long:
        label = label.long()
    
    n_classes = pred.size(1)
    if label.min() < 0 or label.max() >= n_classes:
        print(f"Warning: Invalid label values found. Min: {label.min().item()}, Max: {label.max().item()}, n_classes: {n_classes}")

    loss = lossfn(pred, label)
    error_name = []
    errors_num = 0       
    indices = None
    if task_name == 'cancer':
        if cont:
            pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]
            cancer_grade = labels['cancer_grade']
            code = labels['code']

            error = ((cancer_grade.cpu() == 2) * (pred_classes.cpu() != label.cpu()))
            errors_num = error.sum().item()
            error_name = [code[i] + '\n' for i in range(error.size(0)) if error[i]]
            
    if reduction == 'none':
        if isinstance(loss,tuple):
            loss,indices = loss
        if task_name == 'cancer':
            if indices is not None and len(indices)>0:
                cancer_grade = cancer_grade[indices]

            grade_mask = [ loss_weight if i > 1 else 1 if i == 1  else 1 for i in cancer_grade ]

        else :
            grade_mask = [1]
        multilabel_weight = torch.Tensor(grade_mask).to('cuda')
        loss = multilabel_weight * loss
        loss = loss.mean()

    return loss,errors_num,error_name


def get_optimizer(args,model):
    optimizer = optim.Adam( model.model.parameters(), args.lr)
    return optimizer
