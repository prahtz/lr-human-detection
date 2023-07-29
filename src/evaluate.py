import argparse

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

import utils
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
    test_args = cfg.test
    training_args = cfg.training
    utils.set_seed(random_seed)

    model_module, transforms_fn = load_model_and_transforms(model_args=model_args, checkpoint_path=test_args.checkpoint_path)
    data_module = load_data_module(data_args=data_args, training_args=training_args, test_args=test_args, transforms_fn=transforms_fn)
    logger = TensorBoardLogger(save_dir=training_args.log.run_path, name="", version="test")
    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=logger,
        log_every_n_steps=0,
    )
    trainer.test(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg_path", help="Path of the YAML configuration file.")
    parser.add_argument("--random-seed", help="Manual random seed", default=42, type=int)

    args = parser.parse_args()

    pipeline(args)
