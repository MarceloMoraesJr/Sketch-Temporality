import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import pickle as pkl
import torch
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models import Sketchformer, BlockConfig, PosEmbeddingConfig
from src.data import InputHandler, OutputHandler, QuickDrawDataset
from src.lightning_models import LtSketchReconstruction, LtSketchClassificationMLP


import warnings
warnings.filterwarnings("ignore", message=".*tensorboardX.*")

parser = argparse.ArgumentParser()
#training arguments
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--max_epochs", type=int, default=100)

#general arguments
parser.add_argument("--input_ckpt_path", type=str)
parser.add_argument("--input_results_path", type=str)
parser.add_argument("--output_ckpt_path", type=str)
parser.add_argument("--output_results_path", type=str)

args = parser.parse_args()

with open(args.input_results_path, "rb") as file:
    results = pkl.load(file)

state_dict = torch.load(args.input_ckpt_path, map_location="cpu")['state_dict']

args_lookup = results['args']

seed_everything(args_lookup['seed'], workers=True)

sketchformer = Sketchformer(
    hidden_dim=args_lookup['hidden_dim'],
    num_encoder_layers=args_lookup['num_encoder_layers'],
    num_decoder_layers=args_lookup['num_decoder_layers'],
    decoder_type=args_lookup['decoder_type'],
    block_config=BlockConfig(
        dropout=args_lookup['hidden_dropout']
    ),
    pos_embedding_config=PosEmbeddingConfig(
        pen_state=args_lookup['pen_state'], 
        stroke_embedding=args_lookup['stroke_embedding'], 
        sketch_pos=args_lookup['sketch_pos'],
        stroke_pos=args_lookup['stroke_pos']
    )
)

input_handler = InputHandler(input_relative_coords=results['args']['input_relative_coords'], output_relative_coords=results['args']['output_relative_coords'], autoencoder=True, autoregressive=results['args']['decoder_type'] in ["ar", "ar-enc"])
output_handler = OutputHandler(output_relative_coords=results['args']['output_relative_coords'], autoregressive=results['args']['decoder_type'] in ["ar", "ar-enc"])
model = LtSketchReconstruction(sketchformer, input_handler, output_handler, results['args']['lr'])
model.load_state_dict(state_dict)
model.eval()
model = model.cuda(args.device)
sketchformer = model.sketchformer
sketchformer.decoder = None
sketchformer.ln_decoder = None

train_dataset = QuickDrawDataset("../sketch_representations/data/quickdraw", split='train')
val_dataset = QuickDrawDataset("../sketch_representations/data/quickdraw", split='valid')
test_dataset = QuickDrawDataset("../sketch_representations/data/quickdraw", split='test')
train_loader = DataLoader(train_dataset, batch_size=512, collate_fn=QuickDrawDataset.collate_fn_padd, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=512, collate_fn=QuickDrawDataset.collate_fn_padd, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=512, collate_fn=QuickDrawDataset.collate_fn_padd, num_workers=8)

# Embedding extraction and Dataset building
embeddings, labels = [[],[],[]], [[],[],[]]

for i, loader in enumerate([train_loader, val_loader, test_loader]):
    for batch in tqdm(loader):
        batch = {k: v.cuda(args.device) for k,v in batch.items()}
        model_input = input_handler(batch)
        x_enc = model_input['encoder']
        
        with torch.no_grad():
            h_sketch = sketchformer.encode(x_enc['pos'], x_enc['pos_info'], x_enc['token_id'], x_enc['mask'])

        embeddings[i].append(h_sketch.cpu())
        labels[i].append(x_enc['label'].cpu())

for i in range(3):
    embeddings[i] = torch.cat(embeddings[i], dim=0)
    labels[i] = torch.cat(labels[i])

train_dataset = TensorDataset(embeddings[0], labels[0])
val_dataset = TensorDataset(embeddings[1], labels[1])
test_dataset = TensorDataset(embeddings[2], labels[2])
train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=8)

#MLP Training
classifier_mlp = LtSketchClassificationMLP(input_dim=args_lookup['hidden_dim'], hidden_dim=args_lookup['hidden_dim'] * 2, num_hidden_layers=2, num_classes=345, lr=args.lr)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=args.output_ckpt_path,
    mode="min",
    save_top_k=1,
    filename="best",
    verbose=True
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=15,
    mode="min",
    verbose=True
)

trainer = Trainer(
    max_epochs=args.max_epochs,
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir=args.output_ckpt_path,
    check_val_every_n_epoch=1,
    deterministic=True,
    accelerator="gpu",
    devices=[args.device]
)

trainer.fit(classifier_mlp, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(classifier_mlp, dataloaders=test_loader, ckpt_path="best")

print(classifier_mlp.test_results)

output_results_path = Path(args.output_results_path)
output_results_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_results_path, "wb") as file:
    pkl.dump({"test_results": classifier_mlp.test_results, "pretraining_args": args_lookup, "mlp_args": vars(args)}, file)