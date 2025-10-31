import numpy as np
import lightning as L
import torch 


def set_seed(seed:int):
    L.seed_everything(seed, workers=True)