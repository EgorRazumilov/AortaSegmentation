import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint 
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset

logging.disable(logging.WARNING)
log = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="main")
def main(cfg: OmegaConf) -> None:
    def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    seed_everything(42)

    train_datasets = []
    val_datasets = []
    test_datasets = []
    for key in cfg:
        if 'train_dataset' in key:
            train_datasets.append(hydra.utils.instantiate(cfg[key]))
        elif 'val_dataset' in key:
            val_datasets.append(hydra.utils.instantiate(cfg[key]))
        elif 'test_dataset' in key:
            test_datasets.append(hydra.utils.instantiate(cfg[key]))
        
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)

    train_dataloader = hydra.utils.instantiate(
        cfg.train_dataloader, dataset=train_dataset
    )

    val_dataloader = hydra.utils.instantiate(
        cfg.val_dataloader, 
        dataset=val_dataset, 
    )

    test_dataloader = hydra.utils.instantiate(
        cfg.test_dataloader, 
        dataset=test_dataset,
    )

    pl_module = hydra.utils.instantiate(cfg.pl_module)

    logger = TensorBoardLogger(save_dir=cfg.tb_logs_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_save_path,
        save_top_k=cfg.save_top_k_models,
        monitor=cfg.compare_metric,
        save_weights_only=True,
    )
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        num_nodes=cfg.num_nodes,
        devices=cfg.num_gpus,
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=[
            checkpoint_callback,
        ],
        check_val_every_n_epoch=1,
    )

    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    trainer.fit(
        model=pl_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    trainer.save_checkpoint(Path(cfg.checkpoint_save_path) / "last.ckpt")
    results = trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    with open('metrics.txt', 'w') as f:
        f.write(str(results))

if __name__ == "__main__":
    main()
