import argparse
import math

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import models.utils
import utils
from config.config import get_default_cfg
from datasets.data_modules import load_data_module
from models.model_modules import load_model_module


def pipeline(args):
    cfg_path = args.cfg_path
    random_seed = args.random_seed

    cfg = get_default_cfg()
    cfg.merge_from_file(cfg_path)

    data_args = cfg.dataset
    model_args = cfg.model
    training_args = cfg.training

    utils.init_distributed()
    utils.set_seed(random_seed)
    dist_info = utils.get_distributed_info()

    model, transforms_fn = models.utils.load_model_and_transforms(model_args)

    model_module = load_model_module(
        model=model,
        learning_rate=training_args.lr,
    )

    data_module = load_data_module(
        data_args=data_args, training_args=training_args, transforms_fn=transforms_fn
    )

    callbacks = []
    if training_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=training_args.metric_for_best_model,
                patience=training_args.early_stopping_patience,
                mode=training_args.mode,
            )
        )

    logger = TensorBoardLogger(save_dir=training_args.log.run_path, name="")

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
    parser.add_argument(
        "--random-seed", help="Manual random seed", default=42, type=int
    )

    args = parser.parse_args()

    pipeline(args)
