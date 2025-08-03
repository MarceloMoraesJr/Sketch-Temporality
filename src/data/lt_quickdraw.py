import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from .quickdraw import QuickDrawDataset
from .perturbations_config import PerturbationsConfig

class LtQuickDraw(pl.LightningDataModule):
    def __init__(self, dataset_path, loader_args, perturbations=PerturbationsConfig()):
        super().__init__()
        self.dataset_path = dataset_path
        self.loader_args = loader_args
        self.perturbations = perturbations

    def setup(self, stage=None):
        self.train_dataset = QuickDrawDataset(self.dataset_path, split='train', perturbations=self.perturbations)
        self.val_dataset = QuickDrawDataset(self.dataset_path, split='valid', perturbations=self.perturbations)
        self.test_dataset = QuickDrawDataset(self.dataset_path, split='test', perturbations=self.perturbations)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.loader_args['batch_size'], collate_fn=QuickDrawDataset.collate_fn_padd,
                                           shuffle=True, num_workers=self.loader_args['num_workers'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.loader_args['batch_size'], collate_fn=QuickDrawDataset.collate_fn_padd, 
                                           num_workers=self.loader_args['num_workers'])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.loader_args['batch_size'], collate_fn=QuickDrawDataset.collate_fn_padd, 
                                           num_workers=self.loader_args['num_workers'])
    