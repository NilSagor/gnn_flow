import torch 
import lightning as L


class ResNetLitModule(L.LightningModule):
    def __init__(self):
        super(ResNetLitModule, self).__init__()
        self.save_hyperparameters()

        