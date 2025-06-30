import torch
import pytorch_lightning as pl
from torchmetrics import Metric



class MaskedL2(Metric):
    def __init__(self, pointwise=True):
        super().__init__()
        self.add_state("sum_l2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.pointwise = pointwise

    def update(self, preds, target, mask):
        l2 = ((preds[mask] - target[mask])**2).mean(dim=-1).sum()

        self.sum_l2 += l2
        self.total += mask.sum() if self.pointwise else preds.shape[0]

    def compute(self):
        return self.sum_l2 / self.total
    


class LtSketchReconstruction(pl.LightningModule):
    def __init__(self, sketchformer, input_handler, output_handler, lr=1e-3):
        super().__init__()
        self.sketchformer = sketchformer
        self.input_handler = input_handler
        self.output_handler = output_handler
        self.lr = lr
        self.criterion = torch.nn.MSELoss()
        self.train_l2 = MaskedL2(pointwise=True)
        self.val_l2 = MaskedL2(pointwise=True)
        self.test_l2_point = MaskedL2(pointwise=True)
        self.test_l2_instance = MaskedL2(pointwise=False)

    def forward(self, batch):
        model_input = self.input_handler(batch)
        x_enc, x_dec = model_input['encoder'], model_input['decoder']
        
        h_sketch = self.sketchformer.encode(x_enc['pos'], x_enc['pos_info'], x_enc['token_id'], x_enc['mask'])
        recon_pos = self.sketchformer.decode(h_sketch, x_dec['pos'], x_dec['pos_info'], x_dec['token_id'], mask=x_dec['mask'])
        
        recon_pos = self.output_handler(recon_pos, batch['mask'])

        pos = model_input['ground_truth']
        mask = x_enc['mask']

        return recon_pos, pos, mask

    def training_step(self, batch, batch_idx):
        self.input_handler.set_mode("train")
        self.output_handler.set_mode("train")
        recon_pos, pos, mask = self(batch)
        loss = self.criterion(recon_pos[mask], pos[mask])
        self.train_l2.update(recon_pos, pos, mask)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_loss", self.train_l2.compute(), prog_bar=True)
        self.train_l2.reset()

    def validation_step(self, batch, batch_idx):
        self.input_handler.set_mode("validation")
        self.output_handler.set_mode("validation")
        recon_pos, pos, mask = self(batch)
        self.val_l2.update(recon_pos, pos, mask)

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_l2.compute(), prog_bar=True)
        self.val_l2.reset()

    def test_step(self, batch, batch_idx):
        self.input_handler.set_mode("test")
        self.output_handler.set_mode("test")
        recons_pos, pos, mask = self(batch)
        self.test_l2_point.update(recons_pos, pos, mask)
        self.test_l2_instance.update(recons_pos, pos, mask)

    def on_test_epoch_end(self):
        self.log("test_l2_point", self.test_l2_point.compute(), prog_bar=True)
        self.log("test_l2_instance", self.test_l2_instance.compute(), prog_bar=True)
        self.test_l2_point.reset()
        self.test_l2_instance.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)