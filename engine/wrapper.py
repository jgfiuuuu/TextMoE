from utils.model import TextmoeSeg
from monai.losses import DiceCELoss
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime
import torch.fft
from thop import profile, clever_format
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import itertools

class TextmoeSegWrapper(pl.LightningModule):

    def __init__(self, args):
        
        super(TextmoeSegWrapper, self).__init__()
        
        self.model = TextmoeSeg(args.bert_type, args.vision_type, args.project_dim)

        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False    
        self.ema_decay = 0.99  

        self.lr = args.lr
        self.history = {}
        
        self.loss_fn = DiceCELoss()

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        self.save_hyperparameters()

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(),lr = 0.0003)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =200, eta_min=1e-6)

        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}

    
    def update_ema_variables(self, model, ema_model, global_step,alpha=0.99):

        alpha = min(1 - 1 / (global_step + 1), alpha)  
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
    def forward(self,x):
       
       return self.model.forward(x)
    
    def to_frequency_domain(self,batch):
        return torch.fft.fft2(batch, dim=(-2, -1))

    def to_spatial_domain(self,freq_batch):
        return torch.fft.ifft2(freq_batch, dim=(-2, -1)).real

    def get_high_frequency(self, freq_batch, cutoff=10):
        
        _, _, H, W = freq_batch.shape
       
        freq_indices = torch.meshgrid(torch.arange(H), torch.arange(W))
        freq_indices = torch.stack(freq_indices, dim=-1).to(torch.float32).to(freq_batch.device) 
        
       
        center = torch.tensor([H // 2, W // 2], dtype=torch.float32).to(freq_batch.device)  
        
    
        high_freq_mask = torch.norm(freq_indices - center, dim=-1) > cutoff
        high_freq_batch = torch.where(high_freq_mask.unsqueeze(0).unsqueeze(0), freq_batch, torch.zeros_like(freq_batch))
        
        return high_freq_batch
            
    def frequency_domain_replace(self,data,cutoff=10):
        image, text, dataset_id = data 
        shuffled_indices = torch.randperm(image.size(0))  
        B = image[shuffled_indices]  

        freq1 = self.to_frequency_domain(image)
        freq2 = self.to_frequency_domain(B)
        

        high_freq1 = self.get_high_frequency(freq1, cutoff)
        high_freq2 = self.get_high_frequency(freq2, cutoff)

        swapped_freq1 = freq1 + high_freq2 - self.get_high_frequency(freq1, cutoff)  
        swapped_freq2 = freq2 + high_freq1 - self.get_high_frequency(freq2, cutoff)  

        swapped_batch1 = self.to_spatial_domain(swapped_freq1)
        swapped_batch2 = self.to_spatial_domain(swapped_freq2)
        a1 = (swapped_batch2, text, dataset_id)
        return a1
        
    def sim(self, x, y):
        
        return F.cosine_similarity(x, y, dim=-1)
    
    import itertools

    def distanceloss(self, os_feat):
        losses = []
        
        for scale_feat in os_feat:
            assert scale_feat.dim() == 5, "[B,4,C,H,W]"
            B, num_modality, C, H, W = scale_feat.shape
            
            
            modalities = [scale_feat[:, i] for i in range(2)]  
            
            
            modality_pairs = list(itertools.combinations(modalities, 2))
            
            
            pair_losses = []
            for (mod_i, mod_j) in modality_pairs:
                
                loss = self.sim(mod_i, mod_j)  
                pair_losses.append(loss)
            
            
            scale_loss = torch.stack(pair_losses).mean()  
            losses.append(scale_loss.mean())  

        total_loss = torch.mean(torch.stack(losses))  
        return total_loss
    
    def shared_step(self,batch,batch_idx):
        x, y, flag = batch
        image, text, dataset_id = x
        
        flag = flag.bool()
        if self.trainer.training:   
            if (flag == False).any(): 
                a1 = self.frequency_domain_replace(x)
                preds,os_feat = self(a1)
                with torch.no_grad():
                    ema_preds,ema_os_feat = self.ema_model(x)
                ema_preds = (ema_preds > 0.5).int()  
            else:
                preds,os_feat = self(x)
                ema_preds = y   
            flag_expanded = flag.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            y = torch.where(flag_expanded, y, ema_preds)

            loss = self.loss_fn(preds, y) +  0.1 * self.distanceloss(os_feat)
            self.update_ema_variables(self.model, self.ema_model, self.trainer.global_step, self.ema_decay)
        else:
            preds , os_feat= self(x)
            loss = self.loss_fn(preds, y) 
              
        
        return {'loss': loss, 'preds': preds.detach(), 'y': y.detach()}    
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
        
    def shared_step_end(self,outputs,stage):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](outputs['preds'], outputs['y']).item()
            if stage=="train":
                self.log(name,step_metric,prog_bar=True)
        return outputs["loss"].mean()
        
    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
            
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
            
    def shared_epoch_end(self,outputs,stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage+"_loss").replace('train_','')] for t in outputs])).item()
        dic = {"epoch":epoch,stage+"_loss":stage_loss}
        
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage+"_"+name] = epoch_metric 
        if stage!='test':
            self.history[epoch] = dict(self.history.get(epoch,{}),**dic)    
        return dic 
    
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="train")
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)
        
        #log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor 
        mode = ckpt_cb.mode 
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:   
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor,arr_scores[best_score_idx]),file = sys.stderr)
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        self.log_dict(dic, logger=True)
       
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)