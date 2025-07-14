import torch
import pytorch_lightning as pl
from torchmetrics import Metric

class SegmentationAccuracy(Metric):
    def __init__(self, pixelbased=True):
        super().__init__()
        self.add_state("acc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.pixelbased = pixelbased

    def update(self, preds, targets, mask, stroke_ids):
        if self.pixelbased:
            self.acc += (preds[mask] == targets[mask]).sum()
            self.total += mask.sum()
        else:
            self.total += (stroke_ids.max(dim=1)[0] + 1).sum()
            preds[~mask] = -1

            for stroke in stroke_ids.unique():
                query = mask & (stroke == stroke_ids)
                stroke_total = query.sum(dim=1)
                stroke_acc = ((preds == targets) & query).sum(dim=1) / stroke_total
                self.acc += (stroke_acc > 0.75).sum()
                


    def compute(self):
        return self.acc / self.total


class LtSketchSegmentation(pl.LightningModule):
    def __init__(self, sketchformer, input_handler, lr=1e-3):
        super().__init__()
        self.sketchformer = sketchformer
        self.input_handler = input_handler
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.p_metric = SegmentationAccuracy(pixelbased=True)
        self.c_metric = SegmentationAccuracy(pixelbased=False)

    def forward(self, x):
        x = self.input_handler(x)
        logits = self.sketchformer(x['pos'], x['pos_info'], x['token_id'], x['mask'])
        return logits

    def training_step(self, batch, batch_idx):
        self.input_handler.set_mode("train")
        logits = self(batch)
        loss = self.criterion(logits[batch['mask']], batch['label'][batch['mask']])
        self.log("train_loss", loss, batch_size=batch['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        self.input_handler.set_mode("validation")
        logits = self(batch)
        loss = self.criterion(logits[batch['mask']], batch['label'][batch['mask']])
        self.log("val_loss", loss, prog_bar=True, batch_size=batch['batch_size'])

    def test_step(self, batch, batch_idx):
        self.input_handler.set_mode("test")
        pred = self(batch).argmax(dim=-1)
        self.p_metric.update(pred, batch['label'], batch['mask'], batch['stroke_id'])
        self.c_metric.update(pred, batch['label'], batch['mask'], batch['stroke_id'])

    def on_test_epoch_end(self):
        p_metric = self.p_metric.compute()
        c_metric = self.c_metric.compute()
        self.p_metric.reset()
        self.c_metric.reset()
        self.log('test_p_metric', p_metric, prog_bar=True)
        self.log('test_c_metric', c_metric, prog_bar=True)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)