import torch
from typing import Dict
from abc import abstractmethod
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

class Evaluator:
    @abstractmethod
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        ...

class BinaryClassificationEvaluator(Evaluator):
    def __init__(self) -> None:
        pass
    
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        metrics = {}
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2)
        preds = torch.argmax(logits, dim=-1)

        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds, average='macro')
        metrics['recall'] = recall_score(labels, preds, average='macro')
        metrics['f1'] = f1_score(labels, preds, average='macro')
        try:
            metrics['roc_auc'] = roc_auc_score(one_hot_labels, torch.softmax(logits, dim=-1))
        except ValueError:
            metrics['roc_auc'] = 0.0
        return metrics
    
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        metrics = {}
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2)
        preds = torch.argmax(logits, dim=-1)
        positives_mask = labels == 1

        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['accuracy_positives'] = accuracy_score(labels[positives_mask], preds[positives_mask])
        metrics['accuracy_negatives'] = accuracy_score(labels[~positives_mask], preds[~positives_mask])
        metrics['precision_negatives'], metrics['precision_positives'] = precision_score(labels, preds, average=None, zero_division=1).tolist()
        metrics['recall_negatives'], metrics['recall_positives'] = recall_score(labels, preds, average=None, zero_division=0).tolist()
        metrics['f1'] = f1_score(labels, preds, average='macro')
        try:
            metrics['roc_auc'] = roc_auc_score(one_hot_labels, torch.softmax(logits, dim=-1))
        except ValueError:
            metrics['roc_auc'] = 0.0
        return metrics
