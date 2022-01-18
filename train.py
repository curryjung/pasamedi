import json
import os
from dataloader import TEETHdataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from model import TeethDetectorLit
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def run():

    #Dataloader#
    dataset = TEETHdataset()

    train_dataset, val_dataset = random_split(dataset, [86,20])

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = DataLoader(val_dataset,batch_size= 32, shuffle = True)

    #Trainer and model
    trainer = pl.Trainer(gpus=1,max_epochs=10,log_every_n_steps=1)
    model = TeethDetectorLit(batch_size=32)


    #Train
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)


def auto_scale_test():

    #val_loader = DataLoader(val_dataset,batch_size= 4, shuffle = True)

    #Trainer and model
    trainer = pl.Trainer(auto_scale_batch_size=True,gpus=1,max_epochs=5,log_every_n_steps=1)
    model = TeethDetectorLit(batch_size=86)
    trainer.tune(model)


def backbone_freeze():
    #Dataloader#
    dataset = TEETHdataset()

    train_dataset, val_dataset = random_split(dataset, [86,20])

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = DataLoader(val_dataset,batch_size= 32, shuffle = True)

    #Trainer and model
    trainer = pl.Trainer(gpus=1,max_epochs=10,log_every_n_steps=1)
    model = TeethDetectorLit(batch_size=32)
    
    for param in model.model.backbone.parameters():
        param.requires_grad = False

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    #Train
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader,callbacks=[checkpoint_callback])


def save_model():
    #Dataloader#
    dataset = TEETHdataset()

    train_dataset, val_dataset = random_split(dataset, [86,20])

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = DataLoader(val_dataset,batch_size= 32, shuffle = True)

    #early_stop_condition
    early_stop_callback = EarlyStopping(monitor="val_loss",mode="min",patience=3)
    
    #Trainer and model
    trainer = pl.Trainer(default_root_dir="./",gpus=1,max_epochs=100,log_every_n_steps=1,callbacks=[early_stop_callback])
    model = TeethDetectorLit(batch_size=32)
    
    #Freeze backbone
    for param in model.model.backbone.parameters():
        param.requires_grad = False

    #Train
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)  


def test_infer():
    #Dataloader#
    dataset = TEETHdataset(resize = True)

    train_dataset, val_dataset = random_split(dataset, [86,20])

    # train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
    val_loader = DataLoader(val_dataset,batch_size= 2, shuffle = False)
    #Trainer and model
    model = TeethDetectorLit()

    device = torch.device('cuda')
    model.cuda()
    #model.eval()
    input, target = next(iter(val_loader))

    input = input.cuda()

    with torch.no_grad():
        out = model(input)


    print(out)

    print('!')




if __name__=="__main__":
    #test_infer()
    save_model()


