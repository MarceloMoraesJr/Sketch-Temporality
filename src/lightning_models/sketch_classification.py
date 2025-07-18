import torch
import pytorch_lightning as pl
from torchmetrics import ConfusionMatrix, Accuracy


class LtSketchClassification(pl.LightningModule):
    def __init__(self, sketchformer, input_handler, lr=1e-3, finetuning=False):
        super().__init__()
        self.sketchformer = sketchformer
        self.input_handler = input_handler
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=345)
        self.conf_matrix = ConfusionMatrix(task="multiclass", num_classes=345)

        self.finetuning = finetuning 

    def forward(self, x):
        x = self.input_handler(x)
        logits = self.sketchformer(x['pos'], x['pos_info'], x['token_id'], x['mask'])
        return logits

    def training_step(self, batch, batch_idx):
        self.input_handler.set_mode("train")
        logits = self(batch)
        loss = self.criterion(logits, batch['label'])
        self.log("train_loss", loss, batch_size=batch['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        self.input_handler.set_mode("validation")
        logits = self(batch)
        loss = self.criterion(logits, batch['label'])
        self.log("val_loss", loss, prog_bar=True, batch_size=batch['batch_size'])

    def test_step(self, batch, batch_idx):
        self.input_handler.set_mode("test")
        pred = self(batch).argmax(dim=1)
        self.acc.update(pred, batch['label'])
        self.conf_matrix.update(pred, batch['label'])

    def on_test_epoch_end(self):
        acc = self.acc.compute()
        conf = self.conf_matrix.compute()
        self.conf_matrix.reset()
        self.acc.reset()
        self.log('test_acc', acc, prog_bar=True)
        
        self.test_results = {
            "test_acc": acc.item(),
            "conf_matrix": conf.cpu().numpy()
        }

    def configure_optimizers(self):
        if type(self.lr) == dict:
            return torch.optim.Adam([
                {'params': self.sketchformer.sketchformer.parameters(), 'lr': self.lr['sketchformer']},
                {'params': self.sketchformer.classifier.parameters(), 'lr': self.lr['classifier']}
            ])

        return torch.optim.Adam(self.parameters(), lr=self.lr)