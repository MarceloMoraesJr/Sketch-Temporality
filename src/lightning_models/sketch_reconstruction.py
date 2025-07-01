import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Metric



class MaskedL2(Metric):
    def __init__(self, pointwise=True):
        super().__init__()
        self.add_state("sum_l2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.pointwise = pointwise

    def update(self, preds, targets, mask):
        l2 = ((preds - targets)**2).sum(dim=-1)
        mask = mask.float()
        l2 = l2 * mask

        if not self.pointwise:
            instance_l2 = l2.sum(dim=1)
            instance_l2 = instance_l2 / mask.sum(dim=1).clamp(min=1)
            self.sum_l2 += instance_l2.sum()
            self.total += preds.shape[0]
        else:
            self.sum_l2 += l2.sum() 
            self.total += mask.sum()

    def compute(self):
        return self.sum_l2 / self.total
    

class MaskedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, mask):
        loss = ((preds - targets)**2).sum(dim=-1)
        mask = mask.float()
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + 1e-8)
        return loss
        
    

class LtSketchReconstruction(pl.LightningModule):
    def __init__(self, sketchformer, input_handler, output_handler, lr=1e-3):
        super().__init__()
        self.sketchformer = sketchformer
        self.input_handler = input_handler
        self.output_handler = output_handler
        self.lr = lr
        self.criterion = MaskedL2Loss()
        self.train_l2 = MaskedL2(pointwise=True)
        self.val_l2 = MaskedL2(pointwise=True)
        self.test_l2_point = MaskedL2(pointwise=True)
        self.test_l2_instance = MaskedL2(pointwise=False)

    def forward(self, batch):
        model_input = self.input_handler(batch)
        x_enc, x_dec = model_input['encoder'], model_input['decoder']
        
        h_sketch = self.sketchformer.encode(x_enc['pos'], x_enc['pos_info'], x_enc['token_id'], x_enc['mask'])
        preds = self.sketchformer.decode(h_sketch, x_dec['pos'], x_dec['pos_info'], x_dec['token_id'], mask=x_dec['mask'])
        
        preds = self.output_handler(preds, batch['mask'])

        targets = model_input['targets']
        mask = x_enc['mask']

        return preds, targets, mask

    def training_step(self, batch, batch_idx):
        self.input_handler.set_mode("train")
        self.output_handler.set_mode("train")
        preds, targets, mask = self(batch)
        loss = self.criterion(preds, targets, mask)
        self.train_l2.update(preds, targets, mask)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_loss", self.train_l2.compute(), prog_bar=True)
        self.train_l2.reset()

    def validation_step(self, batch, batch_idx):
        self.input_handler.set_mode("validation")
        self.output_handler.set_mode("validation")
        preds, pos, mask = self(batch)
        self.val_l2.update(preds, pos, mask)

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_l2.compute(), prog_bar=True)
        self.val_l2.reset()

    def test_step(self, batch, batch_idx):
        self.input_handler.set_mode("test")
        self.output_handler.set_mode("test")
        preds, targets, mask = self(batch)
        self.test_l2_point.update(preds, targets, mask)
        self.test_l2_instance.update(preds, targets, mask)

    def on_test_epoch_end(self):
        self.log("test_l2_point", self.test_l2_point.compute(), prog_bar=True)
        self.log("test_l2_instance", self.test_l2_instance.compute(), prog_bar=True)
        self.test_l2_point.reset()
        self.test_l2_instance.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)