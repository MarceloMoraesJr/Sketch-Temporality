import torch
import pytorch_lightning as pl


class LtSketchClassification(pl.LightningModule):
    def __init__(self, sketchformer, input_handler, lr=1e-3):
        super().__init__()
        self.sketchformer = sketchformer
        self.input_handler = input_handler
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.input_handler.seq(x)
        logits = self.sketchformer(x['pos'], x['pos_info'], x['token_id'], x['mask'])
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.criterion(logits, batch['label'])
        self.log("train_loss", loss, batch_size=batch['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.criterion(logits, batch['label'])
        self.log("val_loss", loss, prog_bar=True, batch_size=batch['batch_size'])

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        acc = (logits.argmax(dim=1) == batch['label']).float().mean()
        self.log("test_acc", acc, prog_bar=True, batch_size=batch['batch_size'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)