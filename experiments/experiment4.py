import argparse
import pickle as pkl
import torch
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models import SketchformerClassifier, BlockConfig, PosEmbeddingConfig
from src.data import InputHandler, OutputHandler, LtQuickDraw
from src.lightning_models import LtSketchClassification

import warnings
warnings.filterwarnings("ignore", message=".*tensorboardX.*")

parser = argparse.ArgumentParser()
#training arguments
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--max_epochs", type=int, default=1)

#general arguments
parser.add_argument("--input_ckpt_path", type=str)
parser.add_argument("--input_results_path", type=str)
parser.add_argument("--output_ckpt_path", type=str)
parser.add_argument("--output_results_path", type=str)

#reference arguments (trained from scratch)
parser.add_argument("--reference", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--hidden_dropout", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=8)

parser.add_argument("--num_encoder_layers", type=int, default=4)
parser.add_argument("--hidden_dim", type=int, default=128)

parser.add_argument("--input_relative_coords", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--pen_state", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stroke_embedding", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--sketch_pos", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stroke_pos", action=argparse.BooleanOptionalAction, default=False)


args = parser.parse_args()

if not args.reference:
    with open(args.input_results_path, "rb") as file:
        results = pkl.load(file)

    state_dict = torch.load(args.input_ckpt_path, map_location="cpu")['state_dict']
    
    args_lookup = results['args']
else:
    args_lookup = vars(args)



seed_everything(args_lookup['seed'], workers=True)

sketchformer = SketchformerClassifier(
    hidden_dim=args_lookup['hidden_dim'],
    num_layers=args_lookup['num_encoder_layers'],
    num_classes=345,
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


input_handler = InputHandler(input_relative_coords=args_lookup['input_relative_coords'], autoencoder=False)
model = LtSketchClassification(sketchformer, input_handler, lr=args.lr)
if not args.reference:
    model.sketchformer.load_state_dict(state_dict, strict=False)

datamodule = LtQuickDraw(dataset_path="../sketch_representations/data/quickdraw/",
                        loader_args={"seed": args_lookup['seed'],
                                    "num_workers": args_lookup['num_workers'],
                                    "batch_size": args_lookup['batch_size']})

datamodule.setup()

trainer = Trainer(
    max_epochs=args.max_epochs,
    logger=False,
    enable_checkpointing=False,
    check_val_every_n_epoch=None,
    deterministic=True,
    accelerator="gpu",
    devices=[args.device]
)

trainer.fit(model, train_dataloaders=datamodule.train_dataloader())
trainer.test(model, dataloaders=datamodule.test_dataloader())

output_ckpt_path = Path(args.output_ckpt_path)
output_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

model = model.cpu()
torch.save(model.state_dict(), output_ckpt_path)

output_results_path = Path(args.output_results_path)
output_results_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_results_path, "wb") as file:
    pkl.dump({"test_results": model.test_results, "pretraining_args": args_lookup, "finetuning_args": vars(args) if not args.reference else None}, file)