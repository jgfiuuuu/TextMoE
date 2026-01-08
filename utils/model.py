import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cv2

class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        text_output = output['hidden_states'][-1]
        return text_output
        
            
class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)


        return {"feature":output['hidden_states']}

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:

        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):

        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]
    

class MoeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_experts=2, embed_dim=768, output_text_len=24):
        super(MoeConv, self).__init__()
        self.num_experts = num_experts
        
        
        self.text_project = nn.Sequential(
            nn.Linear(embed_dim, out_channels),  
            nn.GELU(),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.txt_pos = PositionalEncoding(out_channels,max_len=output_text_len)
        self.gate = nn.Linear(in_channels + 2, num_experts)  #  Modality-Indicator
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'conv': nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                'vis_pos': PositionalEncoding(out_channels),
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=out_channels,
                    num_heads=2,
                    batch_first=True
                ),
                'norm': nn.LayerNorm(out_channels)
            }) for _ in range(num_experts)
        ])
        self.scale = nn.Parameter(torch.tensor(1.421),requires_grad=True)

    def forward(self, x, text, dataset_id):
        B, C, H, W = x.shape
        device = x.device

        one_hot_id = F.one_hot(dataset_id, num_classes=2).float()  # [B, 4]
        
        pooled = x.mean(dim=[2, 3])  # [B, C]
        
        gate_input = torch.cat([pooled, one_hot_id], dim=1)  # [B, C+2]
        
        gate_logits = self.gate(gate_input)  # [B, num_experts]   #
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, num_experts]

        txt_proj = self.text_project(text)  # [B, seq_len, in_channels]
        txt = self.txt_pos(txt_proj)
        
        
        expert_outputs = []
        for expert in self.experts:

            expert_vis = expert['conv'](x)  # [B, C_out, H, W]
            vis_seq = expert_vis.permute(0, 2, 3, 1)  # [B, H, W, C_out]
            vis_seq = expert['vis_pos'](vis_seq.view(B, -1, expert_vis.size(1)))  # [B, H*W, C_out]
            '''
            vis_attn, _ = expert['cross_attn'](
                query=expert['norm'](vis_seq),
                key=expert['norm'](vis_seq),
                value=expert['norm'](vis_seq)
            )
            '''
            vis_attn, _ = expert['cross_attn'](
                query=expert['norm'](vis_seq),
                key=txt,
                value=txt
            )

            fused_vis = vis_seq + self.scale * vis_attn
            fused_vis = fused_vis.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, C_out, H, W]
            expert_outputs.append(fused_vis)

        stacked_outputs = torch.stack(expert_outputs, dim=1)  # [B, 2, C_out, H, W]
        gate_weights = gate_weights.view(B, self.num_experts, 1, 1, 1)
        final_output = (gate_weights * stacked_outputs).sum(dim=1)
        
        return final_output,stacked_outputs  # [B, C_out, H, W]
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=2):
        super(UNetDecoderBlock, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.moe_conv1 = MoeConv(in_channels=out_channels * 2, out_channels=out_channels, num_experts=num_experts)
        self.moe_conv2 = MoeConv(in_channels=out_channels, out_channels=out_channels, num_experts=num_experts)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip,text,dataset_id):
       
        x = self.up(x)
       
        x = torch.cat([x, skip], dim=1)

        x,stacked_outputs_1 = self.moe_conv1(x,text,dataset_id)
        x = F.relu(self.bn1(x))

        x,stacked_outputs_2 = self.moe_conv2(x,text,dataset_id)
        x = F.relu(self.bn2(x))
        return x,stacked_outputs_1,stacked_outputs_2


class TextmoeSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(TextmoeSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim) 


        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]
        
        self.block1 = UNetDecoderBlock(feature_dim[0], feature_dim[1], num_experts=2)
        self.block2 = UNetDecoderBlock(feature_dim[1], feature_dim[2], num_experts=2)
        self.block3 = UNetDecoderBlock(feature_dim[2], feature_dim[3], num_experts=2)

        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=3)
    
        
    
    def forward(self, data):

        image, text, dataset_id = data 
        image_output = self.encoder(image)
        image_features = image_output['feature']
        #print(image_features[-1].shape,image_features[-2].shape,image_features[-3].shape,image_features[-4].shape)  #(8, 768, 7, 7) (8, 384, 14, 14) (8, 192, 28, 28) (8, 96, 56, 56)
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            
        
        x,os1,os2 = self.block1(image_features[-1], image_features[-2],text_output,dataset_id)
        x,os3,os4 = self.block2(x, image_features[-3],text_output,dataset_id)
        x1,os5,os6 = self.block3(x, image_features[-4],text_output,dataset_id)
        x = self.decoder1(x1)
          
        out = self.out(x).sigmoid()
        
        os_feat = (os1,os2,os3,os4,os5,os6)

        return out,os_feat
      
