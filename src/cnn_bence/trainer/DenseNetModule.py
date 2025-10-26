import lightning as L
import torchmetrics
import torch 


class DenseNetModule(L.LightningModule):
    def __init__(self, config):
        super(DenseNetModule, self).__init__()
        self.config = config


    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters(), lr=0)
        lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)