import lightning as L


class DenseNetModule(L.LightningModule):
    def __init__(self, config):
        super(DenseNetModule, self).__init__()
        self.config = config