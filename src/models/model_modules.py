from collections import defaultdict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import CatMetric
from torchmetrics.functional import accuracy, auroc, average_precision, f1_score, precision, recall


class PRWClassificationModule(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate=2e-5, loss_fn=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            self.loss_fn = nn.BCEWithLogitsLoss()

        self.model = model
        self.evaluation_step_outputs = defaultdict(lambda: defaultdict(CatMetric))

        self.save_hyperparameters(ignore="model")

    def training_step(self, batch, _):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.loss_fn.__call__(logits, targets)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _, dataloader_idx=0):
        inputs, targets = batch
        logits = self.model(inputs)
        preds = torch.sigmoid(logits)

        self.evaluation_step_outputs[dataloader_idx]["preds"](preds)
        self.evaluation_step_outputs[dataloader_idx]["targets"](targets)

    def test_step(self, batch, _, dataloader_idx=0):
        return self.validation_step(batch, _, dataloader_idx=dataloader_idx)

    def on_validation_epoch_end(self) -> None:
        num_dataloaders = len(self.evaluation_step_outputs.keys())
        all_preds_0 = self.evaluation_step_outputs[0]["preds"].compute()
        all_targets_0 = self.evaluation_step_outputs[0]["targets"].compute()

        positives_mask = all_targets_0 == 1.0
        positives_preds = all_preds_0[positives_mask].unsqueeze(-1)
        positives_targets = all_targets_0[positives_mask].unsqueeze(-1)

        metrics = defaultdict(list)
        for i in range(num_dataloaders):
            if i == 0:
                all_preds = all_preds_0
                all_targets = all_targets_0
            else:
                all_preds = self.evaluation_step_outputs[i]["preds"].compute()
                all_targets = self.evaluation_step_outputs[i]["targets"].compute()
                all_preds = torch.cat([positives_preds, all_preds])
                all_targets = torch.cat([positives_targets, all_targets])

            metrics["val_loss"].append(F.binary_cross_entropy(all_preds, all_targets))
            metrics["accuracy"].append(accuracy(all_preds, all_targets, task="binary"))
            metrics["auroc"].append(auroc(all_preds, all_targets.long(), task="binary"))
            metrics["ap"].append(average_precision(all_preds, all_targets.long(), task="binary"))

            metrics["positives_accuracy"].append(accuracy(positives_preds, positives_targets, task="binary"))
            metrics["positives_precision"].append(precision(all_preds, all_targets, task="binary"))
            metrics["positives_recall"].append(recall(all_preds, all_targets, task="binary"))
            metrics["positives_f1"].append(f1_score(all_preds, all_targets, task="binary"))

            negatives_mask = all_targets == 0.0
            if negatives_mask.any():
                negatives_preds = all_preds[negatives_mask].unsqueeze(-1)
                negatives_targets = all_targets[negatives_mask].unsqueeze(-1)
                metrics["negatives_accuracy"].append(accuracy(negatives_preds, negatives_targets, task="binary"))
                metrics["negatives_precision"].append(precision(1 - all_preds, 1 - all_targets, task="binary"))
                metrics["negatives_recall"].append(recall(1 - all_preds, 1 - all_targets, task="binary"))
                metrics["negatives_f1"].append(f1_score(1 - all_preds, 1 - all_targets, task="binary"))

            self.evaluation_step_outputs[i]["preds"].reset()
            self.evaluation_step_outputs[i]["targets"].reset()

        for metric_name, metric_list in metrics.items():
            t = torch.tensor(metric_list)
            std, mu = torch.std(t, unbiased=False), torch.mean(t)
            self.log(name=f"std_{metric_name}", value=std, on_epoch=True, sync_dist=True)
            self.log(name=f"mu_{metric_name}", value=mu, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
