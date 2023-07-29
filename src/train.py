import argparse
import os

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import utils
from config import config
from config.config import get_default_cfg
from datasets.utils import load_data_module
from models.utils import load_model_and_transforms


def pipeline(args):
    cfg_path = args.cfg_path
    random_seed = args.random_seed
    cfg = get_default_cfg()
    cfg.merge_from_file(cfg_path)
    data_args = cfg.dataset
    model_args = cfg.model
    training_args = cfg.training
    test_args = cfg.test

    utils.init_distributed()
    utils.set_seed(random_seed)
    dist_info = utils.get_distributed_info()

    model_module, transforms_fn = load_model_and_transforms(model_args, learning_rate=training_args.lr)
    data_module = load_data_module(data_args=data_args, training_args=training_args, test_args=test_args, transforms_fn=transforms_fn)

    logger = TensorBoardLogger(save_dir=training_args.log.run_path, name="")
    callbacks = []
    if training_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=training_args.metric_for_best_model,
                patience=training_args.early_stopping_patience,
                mode=training_args.mode,
            )
        )
    callbacks.append(
        ModelCheckpoint(
            monitor="mu_auroc",
            mode="max",
            dirpath=os.path.join(logger.log_dir, "models/"),
            filename="best",
            save_last=True,
        )
    )

    cfg.test.checkpoint_path = os.path.join(logger.log_dir, "models/best.ckpt")
    config.save_cfg(cfg, os.path.join(logger.log_dir, "config/"), "test.yaml")
    trainer = Trainer(
        devices=dist_info["local_world_size"],
        num_nodes=dist_info["num_nodes"],
        logger=logger,
        callbacks=callbacks,
        max_epochs=training_args.num_epochs,
        accumulate_grad_batches=training_args.accumulation_steps,
        log_every_n_steps=0,
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg_path", help="Path of the YAML configuration file.")
    parser.add_argument("--random-seed", help="Manual random seed", default=42, type=int)

    args = parser.parse_args()

    pipeline(args)
