import argparse
import yaml
import pickle as pkl
import torch
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything

from src.models import SketchformerClassifier, BlockConfig, PosEmbeddingConfig
from src.data import InputHandler, LtQuickDraw
from src.lightning_models import LtSketchClassification
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


if not config['reference']:
    with open(config['paths']['model_base_config'], "r") as file:
        model_base_config = yaml.safe_load(file)

    with open(config['paths']['model_config'], "r") as file:
        model_config = yaml.safe_load(file)

    model_config = recursive_merge(model_base_config, model_config)

    state_dict = torch.load(model_config['paths']['checkpoint'] + "/best.ckpt", map_location="cpu")['state_dict']
    
    args_lookup = model_config
else:
    args_lookup = config



seed_everything(args_lookup['training']['seed'], workers=True)

sketchformer = SketchformerClassifier(
    hidden_dim=args_lookup['architecture']['hidden_dim'],
    num_layers=args_lookup['architecture']['num_encoder_layers'],
    num_classes=345,
    block_config=BlockConfig(
        dropout=args_lookup['training']['hidden_dropout']
    ),
    pos_embedding_config=PosEmbeddingConfig(
        pen_state=args_lookup['pe']['pen_state'], 
        stroke_embedding=args_lookup['pe']['stroke_embedding'], 
        sketch_pos=args_lookup['pe']['sketch_pos'],
        stroke_pos=args_lookup['pe']['stroke_pos']
    )
)


input_handler = InputHandler(input_relative_coords=args_lookup['pe']['input_relative_coords'], autoencoder=False)
model = LtSketchClassification(sketchformer, input_handler, lr=args_lookup['training']['lr'])
if not config['reference']:
    model.sketchformer.load_state_dict(state_dict, strict=False)

datamodule = LtQuickDraw(dataset_path="./data/quickdraw/",
                        loader_args={"seed": args_lookup['training']['seed'],
                                    "num_workers": args_lookup['training']['num_workers'],
                                    "batch_size": args_lookup['training']['batch_size']})

datamodule.setup()

trainer = Trainer(
    max_epochs=config['training']['max_epochs'],
    logger=False,
    enable_checkpointing=False,
    check_val_every_n_epoch=None,
    deterministic=True,
    accelerator="gpu",
    devices=[args.device]
)

trainer.fit(model, train_dataloaders=datamodule.train_dataloader())
trainer.test(model, dataloaders=datamodule.test_dataloader())

output_ckpt_path = Path(config['paths']['checkpoint'])
output_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

model = model.cpu()
torch.save(model.state_dict(), output_ckpt_path)

output_results_path = Path(config['paths']['results'])
output_results_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_results_path, "wb") as file:
    pkl.dump({"test_results": model.test_results, "args": vars(args)}, file)