import torch
from typing import Dict
from abc import abstractmethod
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

class Evaluator:
    @abstractmethod
    def __call__(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        ...

class BinaryClassificationEvaluator(Evaluator):
    def __init__(self) -> None:
        pass
    
    def __call__(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        metrics = {}
        preds = preds[preds != -100]
        labels = labels[labels != -100]
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds)
        metrics['recall'] = recall_score(labels, preds)
        metrics['f1'] = f1_score(labels, preds)
        try:
            metrics['roc_auc'] = roc_auc_score(labels, preds)
        except ValueError:
            metrics['roc_auc'] = 1.0

        return metrics
