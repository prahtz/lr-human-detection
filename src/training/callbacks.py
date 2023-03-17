from typing import Dict

class TrainerControl:
    def __init__(self, training_stop=False, training_save=False) -> None:
        self.training_stop = training_stop
        self.training_save = training_save

class Callback:
    def __call__(self, training_log: Dict[str, float], epoch: int) -> TrainerControl:
        ...


class EarlyStoppingCallback(Callback):
    def __init__(self, eval_metric_name: str, higher_is_better=True, patience: int = 0):
        self.eval_metric_name = eval_metric_name
        self.patience = patience
        self.best_metric = 0.0

        if higher_is_better:
            compare = lambda new, old: new > old
        else:
            compare = lambda new, old: new < old
        self.compare = compare
        self.current_patience = 0


    def __call__(self, training_log: Dict[str, float], epoch: int) -> TrainerControl:
        control = TrainerControl()
        eval_metric = training_log[f'eval/{self.eval_metric_name}']
        if epoch == 0 or self.compare(eval_metric, self.best_metric):
            self.best_metric = eval_metric
            self.current_patience = 0
        elif self.patience > 0:
            self.current_patience += 1
            if self.current_patience == self.patience:
                control.training_stop = True
            
        return control
            
            

