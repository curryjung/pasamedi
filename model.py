import torch
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import TEETHdataset

class TeethDetectorLit(pl.LightningModule):
    def __init__(self,batch_size):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained_backbone=True, num_classes=3, aux_loss=True)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size=batch_size

    def forward(self,x):
        output = self.model(x)
        return output['out']

    def training_step(self,batch,batch_idx):
        input, target = batch
        pred = self(input)
        loss = self.criterion(pred,target)
        self.log("Loss/Train",loss,on_step=True,on_epoch=False)
        self.logger.experiment.add_image("input",torch.Tensor.cpu(input[0]),self.current_epoch)
        self.logger.experiment.add_image("target",torch.Tensor.cpu(target[0]),self.current_epoch,dataformats="HW")
        self.logger.experiment.add_image("pred",torch.Tensor.cpu(torch.argmax(torch.exp(pred),dim=1)[0]),self.current_epoch,dataformats="HW")


        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch,batch_idx)
        self.log("val_loss", loss, prog_bar=True,on_step=True,on_epoch=False)
        self.log("val_acc", acc, prog_bar= True,on_step=True,on_epoch=False)

    def test_step(self,batch,batch_idx):
        pass

    def _shared_eval_step(self,batch,batch_indx):
        input, target = batch
        pred = self(input)
        loss = self.criterion(pred,target)
        acc = self.accuracy(pred,target)
        return loss, acc

    def accuracy(self,pred,target):
        pred = torch.argmax(torch.exp(pred),dim=1)
        return (pred == target).float().mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        train_dataset = TEETHdataset()
        return DataLoader(train_dataset,self.batch_size,shuffle=True)
    



