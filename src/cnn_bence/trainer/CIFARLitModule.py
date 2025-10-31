import lightning as L
import torchmetrics
import torch 
import torch.nn as nn
import torch.nn.functional as F

from cnn_bence.models.densenet import DenseNet
import matplotlib.pyplot as plt
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassAveragePrecision
)


act_fn_map = {"relu":nn.ReLU, "gelu": nn.GELU}
act_fn = act_fn_map[kwargs.pop("act_fn", "relu")]

class CifarLitModule(L.LightningModule):
    def __init__(self, config):
        super(CifarLitModule, self).__init__()

        self.save_hyperparameters()
        # self.config = config
        num_classes = config.get("num_classes", 10)

        # self.model = model["model_name"]
        self.loss_fn = nn.CrossEntropyLoss()

        self.acc = MulticlassAccuracy(num_classes=10)
        self.f1 = MulticlassF1Score(num_classes=10)
        self.auroc = MulticlassAUROC(num_classes=10)
        self.ap = MulticlassAveragePrecision(num_classes=10)

        # self.val_acc_values = []


    def forward(self, x):
        # x = x.view(x.size(0), -1)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch         
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch 
        logits = self(x)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        # self.acc.update(preds, y)
        self.log("val_acc", self.acc(preds, y), on_epoch=True, prog_bar=True)
        self.log("val_f1", self.f1(preds, y), on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.auroc(probs, y), on_epoch=True, prog_bar=True)
        self.log("val_ap", self.ap(probs, y), on_epoch=True, prog_bar=True)
        

    # def on_validation_epoch_end(self):

    #     val_epoch_acc = self.val_acc.compute()
    #     self.val_acc_values.append(val_epoch_acc.item())

    #     self.log("val_epoch_acc", val_epoch_acc, prog_bar=True)
    #     print(f"Epoch {self.current_epoch}|Val Acc: {val_epoch_acc:4f}")
    #     self.val_acc.reset()

    #     plt.figure()
    #     plt.plot(self.val_loss_values, label="Val Loss")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.title("Validation Loss over Epoch")
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig(f'Val_loss_epoch_{self.current_epoch}.png')
    #     plt.close()

        
    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters(), lr=1e-3)
        # lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)
        return optimizer