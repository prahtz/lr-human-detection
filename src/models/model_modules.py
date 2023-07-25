import lightning as L
import torch
import torch.nn as nn
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall


class BinaryClassificationModule(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate=2e-5, loss_fn=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            self.loss_fn = nn.BCEWithLogitsLoss()

        self.metrics = {
            "accuracy": Accuracy(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
            "f1": F1Score(task="binary"),
            "auroc": AUROC(task="binary"),
        }
        self.save_hyperparameters(ignore="model")
        self.model = model

    def training_step(self, batch, _):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.loss_fn.__call__(logits, targets)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets)
        preds = torch.sigmoid(logits)

        metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            metrics[metric_name] = metric_fn(preds, targets)
        metrics["negatives_precision"] = self.metrics["precision"](
            1 - preds, 1 - targets
        )
        metrics["negatives_recall"] = self.metrics["recall"](1 - preds, 1 - targets)
        metrics["negatives_f1"] = self.metrics["f1"](1 - preds, 1 - targets)
        metrics["val_loss"] = loss.detach()
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def load_model_module(model: nn.Module, **kwargs):
    module = BinaryClassificationModule(model=model, **kwargs)
    return module
