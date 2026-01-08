import json
import os
import torch
import pandas as pd
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,RandFlipd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
from einops import rearrange, repeat
import matplotlib.pyplot as plt
class QaTa(Dataset):
   
    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224],dataset_id = 1):

        super(QaTa, self).__init__()

        self.mode = mode
        self.dataset_id = dataset_id

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])

        if mode == 'pretrain':
            self.image_list = self.image_list[:int(0.2*len(self.image_list))]
            self.caption_list = self.caption_list[:int(0.2*len(self.caption_list))]

        if mode == 'semi':
            self.labeled_image_list = self.image_list[:int(0.2 * len(self.image_list))]
            self.unlabeled_image_list = self.image_list[int(0.2 * len(self.image_list)):int(0.8 * len(self.image_list))]

            self.labeled_caption_list = self.caption_list[:int(0.2 * len(self.caption_list))]
            self.unlabeled_caption_list = self.caption_list[int(0.2 * len(self.caption_list)):int(0.8 * len(self.caption_list))]

            self.image_list = self.labeled_image_list + self.unlabeled_image_list
            self.caption_list = self.labeled_caption_list + self.unlabeled_caption_list
        elif mode == 'valid':
            self.image_list = self.image_list[int(0.8*len(self.image_list)):]
            self.caption_list = self.caption_list[int(0.8*len(self.caption_list)):]
        else:
            pass   # for mode is 'test'

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):


        trans = self.transform(self.image_size)
        
        
        image = os.path.join(self.root_path,'Images',self.image_list[idx].replace('mask_',''))
        gt = os.path.join(self.root_path,'GTs', self.image_list[idx])
        
        caption = self.caption_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image,gt,token,mask = data['image'],data['gt'],data['token'],data['mask']
        gt = torch.where(gt==255,1,0)
        
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 
        if image.shape[0] == 1:   
            image = repeat(image,'1 h w -> c h w',c=3)
            #gt = repeat(gt,'1 h w -> c h w',c=3)
        if gt.shape[0] == 1:   
            gt = repeat(gt,'1 h w -> c h w',c=3)

        
         
        if self.mode == 'semi' and idx >= len(self.labeled_image_list):
            placeholder_gt = torch.zeros(3, *self.image_size, dtype=torch.int)
            flag = 0
            return ([image, text, self.dataset_id], placeholder_gt,flag)           
        else:   
            flag = 1
            return ([image, text, self.dataset_id], gt,flag)

    def transform(self, image_size=[224, 224]):
        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandRotated(['image', 'gt'], range_x=(-20, 20), prob=1.0, mode=['bicubic', 'nearest']),
                RandFlipd(['image', 'gt'], prob=0.5, spatial_axis=1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),

            ])
        return trans