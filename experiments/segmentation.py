import argparse
import yaml
import pickle as pkl
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models import SketchformerSeg, BlockConfig, PosEmbeddingConfig
from src.data import InputHandler, LtSPG, PerturbationsConfig
from src.lightning_models import LtSketchSegmentation
from src.utils import recursive_merge

import warnings
warnings.filterwarnings("ignore", message=".*tensorboardX.*")

parser = argparse.ArgumentParser()
#training arguments
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

seed_everything(config['training']['seed'], workers=True)

sketchformer = SketchformerSeg(
    hidden_dim=config['architecture']['hidden_dim'],
    num_layers=config['architecture']['num_layers'],
    num_classes=109,
    block_config=BlockConfig(
        dropout=config['training']['hidden_dropout']
    ),
    pos_embedding_config=PosEmbeddingConfig(
        pen_state=config['pe']['pen_state'], 
        stroke_embedding=config['pe']['stroke_embedding'], 
        sketch_pos=config['pe']['sketch_pos'],
        stroke_pos=config['pe']['stroke_pos']
    )
)



input_handler = InputHandler(input_relative_coords=config['pe']['input_relative_coords'], autoencoder=False)
model = LtSketchSegmentation(sketchformer, input_handler, config['training']['lr'])
datamodule = LtSPG(split=config['training']['split'],
                    dataset_path="./data/spg/",
                    loader_args={"seed": config['training']['seed'],
                                "num_workers": config['training']['num_workers'],
                                "batch_size": config['training']['batch_size']},
                    perturbations=PerturbationsConfig(
                        inter_stroke=config['perturbations']['inter_stroke'],
                        intra_stroke=config['perturbations']['intra_stroke'],
                        intra_stroke_rev=config['perturbations']['intra_stroke_rev'],
                        stroke_order=config['perturbations']['stroke_order']
                    ))



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
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir=config['paths']['checkpoint'],
    max_epochs=config['training']['max_epochs'],
    deterministic=True,
    accelerator="gpu",
    devices=[args.device]
)

trainer.fit(model, datamodule=datamodule)
test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

results_path = Path(config['paths']['results'])
results_path.parent.mkdir(parents=True, exist_ok=True)

with open(results_path, "wb") as file:
    pkl.dump({"test_results": test_results[0], "args": vars(args)}, file)