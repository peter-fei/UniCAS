import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

IMAGE_FOLDER_ROOT = "./"
import os  
from PIL import Image  
import torch  
import torchvision.transforms as transforms  
from torch.utils.data import Dataset  
from glob import glob
from einops.layers.torch import Rearrange
import json
import random
import re


class MultiDataSet(Dataset):
    def __init__(self, data,img_batch=800,tasks=['cancer', 'candidiasis','cluecell'],encoder='unicas',task_id=None):
        if isinstance(data,str) and  os.path.isfile(data):
            data = pd.read_csv(data)

        print(len(data))

        if task_id:
            self.data = data[data['task_id'].isin(task_id)]
        else:
            self.data = data 
        self.data = self.data.drop_duplicates()
        print('len data',len(self.data))


        n = len(self.data)

        self.img_batch = img_batch
        self.tasks = tasks
        self.encoder = encoder #if encoder in ('uni','gigapath','conch','1levit','resnet','dino','hibou','dino2','hoptimus','dino_100','dino_100_2','dino_new','dino_1275','pathasst','dino_1275_stainNA') else '256_pt'

        self.file_folder = f'{IMAGE_FOLDER_ROOT}/Pathology_{encoder}_patch'

        # print(len(self.data['fungus']==0))
        self.num_per_cls_list = [ [len(self.data[self.data[task]==0]),len(self.data[self.data[task]==1])] for task in tasks]
        print(self.encoder,tasks,self.num_per_cls_list)
        # exit()
        
        self.columns = self.data.columns.to_list()
        print(self.columns)

        if isinstance(self.tasks,str):
            self.tasks = [tasks]
        # print(self.data.columns.array)
        for i in self.tasks:
            assert i in self.columns, f'task names wrong : get {i} -------- '


        # print(self.data.columns)
        print(f'total_data:{len(self.data)}----------------------------------------------')
        self.columns = self.data.columns.to_list()

        

        

    def __getitem__(self, index):

        #cancer 0	candidiasis 1	cluecell 2

        images_tensor = self.get_imgtensor(index)
        label_dict={}
        

        for i in self.tasks:
            column_index = self.data.columns.get_loc(i)
            idx = self.columns.index(i)
            label = self.data.iloc[index,column_index]
            label_dict[f'{i}_label'] = label
            

        label_dict['name'] = self.data.iloc[index, 0]
        if 'code' in self.data.columns:
            idx = self.data.columns.get_loc('code')
            label_dict['code'] = self.data.iloc[index,idx]
        else:
            label_dict['code'] = self.data.iloc[index, 0].split('/')[-1]
        if 'cancer_grade' in self.data.columns:
            grade_index = self.data.columns.get_loc('cancer_grade')
        else:
            grade_index = self.data.columns.get_loc('cancer')
        label_dict['cancer_grade'] = self.data.iloc[index, grade_index]
        if 'task_id' in self.data.columns:
            idx = self.data.columns.get_loc('task_id')
            label_dict['task_id'] = self.data.iloc[index,idx]
        else:
            label_dict['task_id'] = -1
        label_dict['num_list'] = self.num_per_cls_list
        labels = label_dict

        return images_tensor, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

    def get_imgtensor(self,index):
        folder_path = None
        
        base_folder = self.file_folder
        # print(base_folder)
        folder_path = get_folder_path(self.data.iloc[index, 0],base_folder)

      
        if folder_path is None:
            print(self.data.iloc[index, 0],self.file_folder)
            raise f'Can not Find {self.data.iloc[index, 0]} ----------------------------------'
        if folder_path and os.path.isfile(os.path.join(folder_path,'images.pt')):
            # print(folder_path)
            if os.path.getsize(os.path.join(folder_path,'images.pt')) == 0:
                print(folder_path)

            images_tensor = torch.load(os.path.join(folder_path,'images.pt'),weights_only=True)

           
            images_tensor = images_tensor[:self.img_batch]
           
            return images_tensor
        else:
           raise f' Tensot of {self.data.iloc[index, 0]} not Exist----------------------------------'

        
def get_folder_path(name, floder):

    folder_path = None
    if os.path.isdir(os.path.join(floder, name, 'torch')):
        folder_path = os.path.join(floder, name, 'torch')
    elif os.path.isdir(os.path.join(floder,'yangxing','Torch',name,'torch')):
        folder_path = os.path.join(floder,'yangxing','Torch',name,'torch')
    elif os.path.isdir(os.path.join(floder,'yinxing','Torch',name,'torch')):
        folder_path = os.path.join(floder,'yinxing','Torch',name,'torch')
    elif os.path.isdir(os.path.join(floder, name)):
        folder_path = os.path.join(floder, name)
    elif os.path.isdir(os.path.join(floder,'Torch',name,'torch')):
        folder_path = os.path.join(floder,'Torch',name,'torch')
    elif os.path.isdir(os.path.join(floder,'Data2025',name,'torch')):
        folder_path = os.path.join(floder,'Data2025',name,'torch')

    if folder_path is None:
        print(f'{name} not in {floder}')
        folder_path = None
    return folder_path



