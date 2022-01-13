
import json
import os
from dataloader import TEETHdataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from model import TeethDetectorLit
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

#Tensorboard#
logger = TensorBoardLogger("tb_logs", name = "Teeth Detector_1")

#Dataloader#
dataset = TEETHdataset()
dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)

#Trainer and model


trainer = pl.Trainer(logger=logger,gpus=1,max_epochs=5)
model = TeethDetectorLit()




#Train
trainer.fit(model,train_dataloaders=dataloader)



