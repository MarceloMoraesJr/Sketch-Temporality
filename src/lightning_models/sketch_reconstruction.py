import torch
import pytorch_lightning as pl


class LtSketchReconstruction(pl.LightningModule):
    def __init__(self, sketchformer, input_handler, lr=1e-3):
        super().__init__()
        self.sketchformer = sketchformer
        self.input_handler = input_handler
        self.lr = lr
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        x = self.input_handler.seq2seq(x, self.sketchformer.decoder_type in ["ar", "ar-enc"])
        x_enc, x_dec = x['encoder'], x['decoder']
        
        h_sketch = self.sketchformer.encode(x_enc['pos'], x_enc['pos_info'], x_enc['token_id'], x_enc['mask'])
        
        if self.sketchformer.decoder_type in ["ar", "ar-enc"]:
            if self.sketchformer.training:
                pred,_ = self.sketchformer.decode(h_sketch, x_dec['pos'], x_dec['pos_info'], x_dec['token_id'], mask=x_dec['mask'])
                pred = pred[:, :-1]
            else:
                pred = None
                cache = None
                for _ in range(x_dec['batch_length']):
                    ar_x_dec = self.input_handler.ar_prediction(x_dec, pred)
                    pred,cache = self.sketchformer.decode(h_sketch, ar_x_dec['pos'], ar_x_dec['pos_info'], ar_x_dec['token_id'], cache)
        else:
            pred,_ = self.sketchformer.decode(h_sketch, x_dec['pos'], x_dec['pos_info'], x_dec['token_id'], mask=x_dec['mask'])

        return pred

    def training_step(self, batch, batch_idx):
        recon_pos = self(batch)
        pos = batch['pos']
        mask = batch['mask']
        loss = self.criterion(recon_pos[mask], pos[mask])
        self.log("train_loss", loss * mask.sum(), batch_size=mask.sum())
        return loss

    def validation_step(self, batch, batch_idx):
        recon_pos = self(batch)
        pos = batch['pos']
        mask = batch['mask']
        loss = self.criterion(recon_pos[mask], pos[mask])
        self.log("val_loss", loss * mask.sum(), prog_bar=True, batch_size=mask.sum())

    def test_step(self, batch, batch_idx):
        recons_pos = self(batch)
        pos = batch['pos']
        mask = batch['mask']
        
        recons_pos[mask] = 0.0
        l2 = ((recons_pos - pos)**2).float().mean(dim=-1).sum()

        self.log("test_l2_pointwise", l2, prog_bar=True, batch_size=batch['mask'].sum())
        self.log("test_l2_instancewise", l2, prog_bar=True, batch_size=batch['batch_size'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)