import os

from yacs.config import CfgNode as CN


class DatasetArgs(CN):
    root_path = "data/PRW"
    dataset_name = "prw-classification"
    positives_relative_path = "positives"
    negatives_relative_path = "negatives"


class ModelArgs(CN):
    name = "customnet"


class LogArgs(CN):
    run_path = "runs/default-run/"


class TrainingArgs(CN):
    log = LogArgs()
    lr = 5.0e-5
    num_workers = 4
    train_batch_size = 32
    eval_batch_size = 32
    num_epochs = 10
    accumulation_steps = 1
    metric_for_best_model = "mu_auroc"
    mode = "max"
    early_stopping_patience = 0
    eval_num_repetitions = 1


class TestArgs(CN):
    checkpoint_path = "path/to/checkpoint"
    test_batch_size = 32
    num_workers = 4
    test_num_repetitions = 33


class RootArgs(CN):
    dataset = DatasetArgs()
    model = ModelArgs()
    training = TrainingArgs()
    test = TestArgs()


def get_default_cfg() -> RootArgs:
    return RootArgs()


def save_cfg(cfg: RootArgs, dirpath: str, filename: str):
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, filename), "w") as f:
        f.write(cfg.dump())


def save_default_cfg():
    with open("src/config/default.yaml", "w") as f:
        f.write(RootArgs().dump())


if __name__ == "__main__":
    save_default_cfg()
