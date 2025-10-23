import lightning as L
from torchvision.datasets.cifar import CIFAR10
from cnn_bence.utils.paths import get_project_root
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader

class CIFAR10Module(L.LightningDataModule):
    def __init__(self, config):
        super(CIFAR10Module, self).__init__()
        self.batch_size = config["data"]["batch_size"]
        self.train_transform  = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((32,32)),
            T.ToTensor(),
            T.Normalize((0.49, 0.48, 0.44), (0.25, 0.24, 0.26))
        ])
        self.test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.49, 0.48, 0.44), (0.25, 0.24, 0.26))
        ])

        self.data_dir = get_project_root()/"data"

        self.train = None
        self.val = None
        self.test = None



    def prepare_data(self):
        CIFAR10(root=str(self.data_dir), train=True, download=True)
        CIFAR10(root=str(self.data_dir), train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_train = CIFAR10(str(self.data_dir), train=True, transform=self.train_transform, download=False)
            self.train, self.val = random_split(full_train, [45000, 5000])

        if stage=="test" or stage is None:
            self.test = CIFAR10(str(self.data_dir), train=False, transform=self.test_transform, download=False)
            

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    

if __name__ == "__main__":
    print("Data module starting...")
    config={"data":{"batch_size": 32}}
    dm = CIFAR10Module(config)
    dm.prepare_data()
    dm.setup(stage="fit")
    dl=dm.train_dataloader()
    x,y = next(iter(dl))
    print("Data shape:", x.shape, " ", y.shape)
