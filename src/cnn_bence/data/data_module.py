import lightning as L
from torchvision.datasets.cifar import CIFAR10
from cnn_bence.utils.paths import get_project_root

class CIFAR10Module(L.LightningDataModule):
    def __init__(self, batch_size=32):
        super(CIFAR10Module, self).__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        CIFAR10(root=get_project_root()/"data", train=True, download=True)
        CIFAR10(root=get_project_root()/"data", train=False, download=True)

    def setup(self, stage=None):
        ...

    def train_dataloader(self):
        return ...
    

if __name__ == "main":
    dm = CIFAR10Module()
    dm.prepare_data()
