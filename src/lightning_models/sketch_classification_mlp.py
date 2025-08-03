import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import ConfusionMatrix, Accuracy


class LtSketchClassificationMLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, num_classes, lr=1e-3):
        super().__init__()

        layers = []
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.layers = nn.ModuleList(layers)

        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=345)
        self.conf_matrix = ConfusionMatrix(task="multiclass", num_classes=345)

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        return x

    def training_step(self, batch, batch_idx):
        logits = self(batch[0])
        loss = self.criterion(logits, batch[1])
        self.log("train_loss", loss, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch[0])
        loss = self.criterion(logits, batch[1])
        self.log("val_loss", loss, prog_bar=True, batch_size=batch[0].shape[0])

    def test_step(self, batch, batch_idx):
        pred = self(batch[0]).argmax(dim=1)
        self.acc.update(pred, batch[1])
        self.conf_matrix.update(pred, batch[1])

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)