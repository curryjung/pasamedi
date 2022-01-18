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
from torchvision.utils import save_image
import cv2

weights_path = "lightning_logs/version_1/checkpoints/epoch=31-step=95.ckpt"
model = TeethDetectorLit(batch_size=32).load_from_checkpoint(weights_path,batch_size=32)

device = torch.device("cuda")
model.eval()
model.to(device)


test_dir = ['./data/sample1','./data/sample2','./data/sample3','./data/sample4']
results_path = os.path.join(test_dir[3],'results')

if not os.path.exists(results_path):
    os.mkdir(results_path)

dataset = TEETHdataset(transform_type= "resize",mode = "test", test_dir = test_dir[3])
dataloader = DataLoader(dataset,batch_size=1,shuffle=False)

H, W = 1080, 1920
# trans = torch.nn.Sequential(
#     transforms.Resize(size=torch.tensor([H, W]))
#     transforms.ToPILImage()
# )
trans = transforms.ToPILImage()


for i, batch in enumerate(dataloader):
    input = batch[0].to(device)
    pred = model(input) #1x3x224x224

    pred = torch.Tensor.cpu(pred) 
    pred = torch.argmax(pred,dim=1) #1x224x224 
    pred = pred.type(torch.LongTensor) 
    # pred = pred.expand(3,-1,-1)/2  #3x224x224

    # c, h, w = pred.shape[:3]
    # pred_viz = torch.ones(1,h, w).type(torch.float64)
    # pred_viz[pred==0] = torch.tensor([0.1, 0.1, 0.1])
    # pred_viz[pred==1] = torch.tensor([1.0, 0., 0.])
    # pred_viz[pred==2] = torch.tensor([0.0, 1., 0])
    # pred_viz = pred_viz.permute(2,0,1)
    c, h, w = pred.shape[:3]
    pred_viz = torch.ones(1,h, w).type(torch.float32)
    pred_viz[pred==0] = torch.tensor([0.0])
    pred_viz[pred==1] = torch.tensor([0.5])
    pred_viz[pred==2] = torch.tensor([1.0])
    pred_viz=pred_viz.expand(3,-1,-1)
    #pred_viz = pred_viz.permute(2,0,1)
    
    
    viz = torch.cat([input.cpu()[0], pred_viz.detach().cpu()], dim=2)

    img = trans(viz)
    img.save(os.path.join(results_path,os.path.basename(dataset.test_list[i])))

