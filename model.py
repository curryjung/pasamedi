import torch
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn as nn

class TeethDetectorLit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained_backbone=True, num_classes=2, aux_loss=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x):
        output = self.model(x)
        return output['out']

    def training_step(self,batch,batch_idx):
        input, target = batch
        pred = self(input)
        loss = self.criterion(pred,target)
        self.logger.experiment.add_scalar("Loss/Train",loss,self.current_epoch)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    

    



