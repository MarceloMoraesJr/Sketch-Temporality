from src.data import QuickDrawDataset, InputHandler
from torch.utils.data import DataLoader

dataset = QuickDrawDataset("./data/quickdraw", split='train', relative_coords=False)
dataloader = DataLoader(dataset, batch_size=512, collate_fn=QuickDrawDataset.collate_fn_padd)

batch = next(iter(dataloader))
batch = InputHandler().seq(batch)
print(batch.keys())
