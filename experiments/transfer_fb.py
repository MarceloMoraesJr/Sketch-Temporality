import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import yaml
import pickle as pkl
import torch
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models import Sketchformer, BlockConfig, PosEmbeddingConfig, DecoderConfig, ModificationsConfig
from src.data import InputHandler, OutputHandler, QuickDrawDataset
from src.lightning_models import LtSketchReconstruction, LtSketchClassificationMLP
from src.utils import recursive_merge


import warnings
warnings.filterwarnings("ignore", message=".*tensorboardX.*")

parser = argparse.ArgumentParser()
parser.add_argument("--base_config", type=str, default=None)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

base_config = {}
if args.base_config is not None:
    with open(args.base_config, "r") as file:
        base_config = yaml.safe_load(file)

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

config = recursive_merge(base_config, config)

with open(config['paths']['model_base_config'], "r") as file:
    model_base_config = yaml.safe_load(file)

with open(config['paths']['model_config'], "r") as file:
    model_config = yaml.safe_load(file)

model_config = recursive_merge(model_base_config, model_config)

state_dict = torch.load(model_config['paths']['checkpoint'] + "/best.ckpt", map_location="cpu")['state_dict']



sketchformer = Sketchformer(
    hidden_dim=model_config['architecture']['hidden_dim'],
    num_encoder_layers=model_config['architecture']['num_encoder_layers'],
    num_decoder_layers=model_config['architecture']['num_decoder_layers'],
    decoder_config=DecoderConfig(
        autoregressive=model_config['architecture']['decoder']['autoregressive'],
        cross_attn=model_config['architecture']['decoder']['cross_attn'],
        condition_first=model_config['architecture']['decoder']['condition_first'],
        modifications_config=ModificationsConfig(
            condition_every=model_config['architecture']['decoder']['condition_every'],
            replace_cross_attn=model_config['architecture']['decoder']['replace_cross_attn'],
            remove_self_attn=model_config['architecture']['decoder']['remove_self_attn'],
            replace_self_attn=model_config['architecture']['decoder']['replace_self_attn']
        )
    ),
    block_config=BlockConfig(
        dropout=model_config['training']['hidden_dropout']
    ),
    pos_embedding_config=PosEmbeddingConfig(
        pen_state=model_config['pe']['pen_state'], 
        stroke_embedding=model_config['pe']['stroke_embedding'], 
        sketch_pos=model_config['pe']['sketch_pos'],
        stroke_pos=model_config['pe']['stroke_pos']
    )
)

input_handler = InputHandler(input_relative_coords=model_config['pe']['input_relative_coords'], output_relative_coords=model_config['pe']['output_relative_coords'], autoencoder=True, autoregressive=model_config['architecture']['decoder']['autoregressive'])
output_handler = OutputHandler(output_relative_coords=model_config['pe']['output_relative_coords'], autoregressive=model_config['architecture']['decoder']['autoregressive'])
model = LtSketchReconstruction(sketchformer, input_handler, output_handler, model_config['training']['lr'])
model.load_state_dict(state_dict)
model.eval()
model = model.cuda(args.device)
sketchformer = model.sketchformer
sketchformer.decoder = None
sketchformer.ln_decoder = None

train_dataset = QuickDrawDataset("./data/quickdraw", split='train')
val_dataset = QuickDrawDataset("./data/quickdraw", split='valid')
test_dataset = QuickDrawDataset("./data/quickdraw", split='test')
train_loader = DataLoader(train_dataset, batch_size=model_config['training']['batch_size'], collate_fn=QuickDrawDataset.collate_fn_padd, num_workers=model_config['training']['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=model_config['training']['batch_size'], collate_fn=QuickDrawDataset.collate_fn_padd, num_workers=model_config['training']['num_workers'])
test_loader = DataLoader(test_dataset, batch_size=model_config['training']['batch_size'], collate_fn=QuickDrawDataset.collate_fn_padd, num_workers=model_config['training']['num_workers'])

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

del sketchformer
del model

#Seeding only after initializing sketchformer due to different decoders moving generator state
seed_everything(model_config['training']['seed'], workers=True)

for i in range(3):
    embeddings[i] = torch.cat(embeddings[i], dim=0)
    labels[i] = torch.cat(labels[i])

train_dataset = TensorDataset(embeddings[0], labels[0])
val_dataset = TensorDataset(embeddings[1], labels[1])
test_dataset = TensorDataset(embeddings[2], labels[2])
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'])
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'])

#MLP Training
classifier_mlp = LtSketchClassificationMLP(input_dim=model_config['architecture']['hidden_dim'], hidden_dim=model_config['architecture']['hidden_dim'] * 2, num_hidden_layers=2, num_classes=345, lr=config['training']['lr'])

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=config['paths']['checkpoint'],
    mode="min",
    save_top_k=1,
    filename="best",
    verbose=True
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=config['training']['patience'],
    mode="min",
    verbose=True
)

trainer = Trainer(
    max_epochs=config['training']['max_epochs'],
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir=config['paths']['checkpoint'],
    check_val_every_n_epoch=1,
    deterministic=True,
    accelerator="gpu",
    devices=[args.device]
)

trainer.fit(classifier_mlp, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(classifier_mlp, dataloaders=test_loader, ckpt_path="best")

output_results_path = Path(config['paths']['results'])
output_results_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_results_path, "wb") as file:
    pkl.dump({"test_results": classifier_mlp.test_results, "args": vars(args)}, file)