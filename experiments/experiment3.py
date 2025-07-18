import argparse
import pickle as pkl
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models import SketchformerSeg, BlockConfig, PosEmbeddingConfig
from src.data import InputHandler, LtSPG, PerturbationsConfig
from src.lightning_models import LtSketchSegmentation

import warnings
warnings.filterwarnings("ignore", message=".*tensorboardX.*")

parser = argparse.ArgumentParser()
#training arguments
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden_dropout", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--max_epochs", type=int, default=150)
parser.add_argument("--patience", type=int, default=15)

#architecture arguments
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--hidden_dim", type=int, default=128)

#positional embedding arguments
parser.add_argument("--relative_coords", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--pen_state", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stroke_embedding", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--sketch_pos", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stroke_pos", action=argparse.BooleanOptionalAction, default=False)

#perturbation arguments
parser.add_argument("--inter_stroke", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--intra_stroke", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--intra_stroke_rev", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--stroke_order", action=argparse.BooleanOptionalAction, default=False)

#general arguments
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--results_path", type=str)

args = parser.parse_args()

seed_everything(args.seed, workers=True)

sketchformer = SketchformerSeg(
    hidden_dim=args.hidden_dim,
    num_layers=args.num_layers,
    num_classes=109,
    block_config=BlockConfig(
        dropout=args.hidden_dropout
    ),
    pos_embedding_config=PosEmbeddingConfig(
        pen_state=args.pen_state, 
        stroke_embedding=args.stroke_embedding, 
        sketch_pos=args.sketch_pos,
        stroke_pos=args.stroke_pos
    )
)



input_handler = InputHandler(input_relative_coords=args.relative_coords, autoencoder=False)
model = LtSketchSegmentation(sketchformer, input_handler, args.lr)
datamodule = LtSPG(split=args.split,
                    dataset_path="./data/spg/",
                    loader_args={"seed": args.seed,
                                "num_workers": args.num_workers,
                                "batch_size": args.batch_size},
                    perturbations=PerturbationsConfig(
                        inter_stroke=args.inter_stroke,
                        intra_stroke=args.intra_stroke,
                        intra_stroke_rev=args.intra_stroke_rev,
                        stroke_order=args.stroke_order
                    ))



checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=args.ckpt_path,
    mode="min",
    save_top_k=1,
    filename="best",
    verbose=True
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=args.patience,
    mode="min",
    verbose=True
)

trainer = Trainer(
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir=args.ckpt_path,
    max_epochs=args.max_epochs,
    deterministic=True,
    accelerator="gpu",
    devices=[args.device]
)

trainer.fit(model, datamodule=datamodule)
test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

results_path = Path(args.results_path)
results_path.parent.mkdir(parents=True, exist_ok=True)

with open(results_path, "wb") as file:
    pkl.dump({"test_results": test_results[0], "args": vars(args)}, file)