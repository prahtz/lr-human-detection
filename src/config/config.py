from yacs.config import CfgNode as CN
import os

class DatasetArgs(CN):
    root_path = 'data/'
    positives_relative_path = 'positives'
    negatives_relative_path = 'negatives'

class ModelArgs(CN):
    name = 'customnet'

class LogArgs(CN):
    run_path = 'runs/default-run/'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.path.exists(self.run_path):
            print('WARNING: the provided run path already exists, please change it to prevent the override of old files.')
        self.models_path = os.path.join(self.run_path, 'models')
        self.log_path = os.path.join(self.run_path, 'logs')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

class TrainingArgs(CN):
    log = LogArgs()
    lr = 5.0e-5
    num_workers = 4
    train_batch_size = 32
    eval_batch_size = 32
    test_batch_size = 32
    num_epochs = 10
    accumulation_steps = 1

class RootArgs(CN):
    dataset = DatasetArgs()
    model = ModelArgs()
    training = TrainingArgs()

def get_default_cfg() -> RootArgs:
    return RootArgs()

def save_default_cfg():
    with open('src/config/default.yaml', 'w') as f:
        f.write(RootArgs().dump())

if __name__ == '__main__':
    save_default_cfg()
