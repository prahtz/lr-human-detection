import argparse
import os

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import utils
from config import config
from config.config import get_default_cfg
from datasets.data_modules import load_data_module
from datasets.prw import PRWClassificationFromSubtraction
from models.utils import load_model_and_transforms


def train(args):
    cfg_path = args.cfg_path
    random_seed = args.random_seed
    ckpt_path = args.ckpt_path
    cfg = get_default_cfg()
    cfg.merge_from_file(cfg_path)
    data_args = cfg.dataset
    model_args = cfg.model
    training_args = cfg.training
    test_args = cfg.test

    utils.init_distributed()
    utils.set_seed(random_seed)
    dist_info = utils.get_distributed_info()

    model_module, train_transforms_fn, eval_transforms_fn, label_transforms_fn = load_model_and_transforms(
        model_args,
        learning_rate=training_args.lr,
    )
    data_module = load_data_module(
        data_args=data_args,
        training_args=training_args,
        test_args=test_args,
        train_transforms_fn=train_transforms_fn,
        eval_transforms_fn=eval_transforms_fn,
        label_transforms_fn=label_transforms_fn,
    )

    if data_args.dataset_name == "prw-classification-from-subtraction":
        temp_dataset = PRWClassificationFromSubtraction(data_args.root_path, "train")
        pos_weight = temp_dataset.num_negatives / temp_dataset.num_positives
        del temp_dataset
        model_module.loss_fn.pos_weight = torch.tensor(pos_weight)

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
            monitor=training_args.metric_for_best_model,
            mode=training_args.mode,
            dirpath=os.path.join(logger.log_dir, "models/"),
            filename="best",
            save_last=True,
        )
    )
    cfg.test.checkpoint_path = os.path.join(logger.log_dir, "models/best.ckpt")
    config.save_cfg(cfg, os.path.join(logger.log_dir, "config/"), "test.yaml")
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=dist_info["local_world_size"],
        num_nodes=dist_info["num_nodes"],
        logger=logger,
        callbacks=callbacks,
        max_epochs=training_args.num_epochs,
        accumulate_grad_batches=training_args.accumulation_steps,
        log_every_n_steps=0,
    )

    trainer.fit(model_module, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg_path", help="Path of the YAML configuration file.")
    parser.add_argument("--random-seed", help="Manual random seed", default=42, type=int)
    parser.add_argument("--ckpt-path", help="Resume training from the specified checkpoint", default=None, type=str)

    args = parser.parse_args()

    train(args)
