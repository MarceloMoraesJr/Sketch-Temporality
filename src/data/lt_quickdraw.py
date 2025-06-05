import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from src.data import QuickDrawDataset

class LtQuickDraw(pl.LightningDataModule):
    def __init__(self, dataset_path, dataset_args, loader_args):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_args = dataset_args
        self.loader_args = loader_args

    def setup(self, stage=None):
        self.train_dataset = QuickDrawDataset(self.dataset_path, split='train', relative_coords=self.dataset_args['relative_coords'])
        self.val_dataset = QuickDrawDataset(self.dataset_path, split='valid', relative_coords=self.dataset_args['relative_coords'])
        self.test_dataset = QuickDrawDataset(self.dataset_path, split='test', relative_coords=self.dataset_args['relative_coords'])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.loader_args['batch_size'], collate_fn=QuickDrawDataset.collate_fn_padd,
                                           shuffle=True, num_workers=self.loader_args['num_workers'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.loader_args['batch_size'], collate_fn=QuickDrawDataset.collate_fn_padd, 
                                           num_workers=self.loader_args['num_workers'])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.loader_args['batch_size'], collate_fn=QuickDrawDataset.collate_fn_padd, 
                                           num_workers=self.loader_args['num_workers'])
    