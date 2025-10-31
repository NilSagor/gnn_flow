import lightning as L

from lightning.pytorch.loggers import CSVLogger
from cnn_bence.utils.seed import set_seed





def train_with_seed(config, seed:int):
    set_seed(42)

    logger = CSVLogger(save_dir=config["log_dir"], name=f"{config['model_name']}")

    model = ...

    trainer = L.Trainer(
        max_epochs= 10,
        accelerator="auto",
        logger = logger,
        log_every_n_steps= 10
    )

    trainer.fit(model)
    trainer.test(model)
    return 0.5