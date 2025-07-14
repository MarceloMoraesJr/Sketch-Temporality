import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from src.data import SPGDataset

class LtSPG(pl.LightningDataModule):
    def __init__(self, split, dataset_path, loader_args):
        super().__init__()
        self.split = split
        self.dataset_path = dataset_path
        self.loader_args = loader_args

    def setup(self, stage=None):
        self.train_dataset = SPGDataset(self.dataset_path, split=f'train-{self.split}')
        self.val_dataset = SPGDataset(self.dataset_path, split=f'valid-{self.split}')
        self.test_dataset = SPGDataset(self.dataset_path, split=f'test-{self.split}')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.loader_args['batch_size'], collate_fn=SPGDataset.collate_fn_padd,
                                           shuffle=True, num_workers=self.loader_args['num_workers'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.loader_args['batch_size'], collate_fn=SPGDataset.collate_fn_padd, 
                                           num_workers=self.loader_args['num_workers'])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.loader_args['batch_size'], collate_fn=SPGDataset.collate_fn_padd, 
                                           num_workers=self.loader_args['num_workers'])
    