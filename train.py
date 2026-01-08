import torch
from torch.utils.data import DataLoader, random_split
from utils.dataset import QaTa
import utils.config as config
from torch.optim import lr_scheduler
from engine.wrapper import TextmoeSegWrapper

import pytorch_lightning as pl    
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import random
import numpy as np
import os

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def get_parser():
    parser = argparse.ArgumentParser(
        description='Textmoe')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg


class ShuffledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        random.shuffle(self.indices)  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        shuffled_idx = self.indices[idx]  
        return self.dataset[shuffled_idx]

if __name__ == '__main__':
    set_seed(0)
    args = get_parser()
    print("cuda:", torch.cuda.is_available())
    # ------------------------------------
    print("Preparing labeled data...")

    ds_labeled_0 = QaTa(csv_path=args.train_csv_path_0,
                      root_path=args.train_root_path_0,
                      tokenizer=args.bert_type,
                      image_size=args.image_size,
                      mode='pretrain',
                      dataset_id=0)
    ds_valid_0 = QaTa(csv_path=args.train_csv_path_0,
                      root_path=args.train_root_path_0,
                      tokenizer=args.bert_type,
                      image_size=args.image_size,
                      mode='valid',
                      dataset_id=0)
    
    ds_labeled_1 = QaTa(csv_path=args.train_csv_path_1,
                      root_path=args.train_root_path_1,
                      tokenizer=args.bert_type,
                      image_size=args.image_size,
                      mode='pretrain',
                      dataset_id=1)
    ds_valid_1 = QaTa(csv_path=args.train_csv_path_1,
                      root_path=args.train_root_path_1,
                      tokenizer=args.bert_type,
                      image_size=args.image_size,
                      mode='valid',
                      dataset_id=1)
    
    
    

    ds_labeled_1_expanded = torch.utils.data.ConcatDataset([ds_labeled_1] * 3)
    

    combined_labeled_dataset = torch.utils.data.ConcatDataset([ds_labeled_0, ds_labeled_1_expanded])
    combined_valid_dataset = torch.utils.data.ConcatDataset([ds_valid_0, ds_valid_1])

    shuffled_labeled_dataset = ShuffledDataset(combined_labeled_dataset)
    shuffled_valid_dataset = ShuffledDataset(combined_valid_dataset)

    dl_labeled = DataLoader(shuffled_labeled_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size)
    dl_valid = DataLoader(shuffled_valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size)

    print("Start pre-training with labeled data...")
    model = TextmoeSegWrapper(args)

    model_ckpt_pre = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
    )
    early_stopping_pre = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )

    trainer_pre = pl.Trainer(
        logger=True,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.device,
        callbacks=[model_ckpt_pre, early_stopping_pre],
        enable_progress_bar=False,
    )
    trainer_pre.fit(model, dl_labeled, dl_valid)
    print("Pre-training complete.")

    # ------------------------------------
    print("Preparing for semi-supervised training...")
    model = TextmoeSegWrapper(args)

    checkpoint = torch.load('./save_model/medseg.ckpt', map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint, strict=True)


    ds_semi_0 = QaTa(csv_path=args.train_csv_path_0,
                      root_path=args.train_root_path_0,
                      tokenizer=args.bert_type,
                      image_size=args.image_size,
                      mode='semi',
                      dataset_id=0)
    ds_semi_1 = QaTa(csv_path=args.train_csv_path_1,
                      root_path=args.train_root_path_1,
                      tokenizer=args.bert_type,
                      image_size=args.image_size,
                      mode='semi',
                      dataset_id=1)
                      
    ds_semi_1_expanded = torch.utils.data.ConcatDataset([ds_semi_1] * 3)
    
    
    combined_semi_dataset = torch.utils.data.ConcatDataset([ds_semi_0, ds_semi_1_expanded])
    shuffled_semi_dataset = ShuffledDataset(combined_semi_dataset)

    dl_semi = DataLoader(shuffled_semi_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size)

    model_ckpt_semi = ModelCheckpoint(
        dirpath='./semi_supervised',
        filename='semi_supervised',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
    )
    early_stopping_semi = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )

    trainer_semi = pl.Trainer(
        logger=True,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.device,
        callbacks=[model_ckpt_semi, early_stopping_semi],
        enable_progress_bar=False,
    )
    print("Start semi-supervised training...")
    trainer_semi.fit(model, dl_semi, dl_valid)
    print("Semi-supervised training complete.")

