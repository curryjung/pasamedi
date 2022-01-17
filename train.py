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
    print(5)

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
        param.requires.grad = False

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




    

    pass




if __name__=="__main__":
    #test_infer()
    run()


