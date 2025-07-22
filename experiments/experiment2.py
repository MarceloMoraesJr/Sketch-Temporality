import argparse
import pickle as pkl
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models import Sketchformer, BlockConfig, PosEmbeddingConfig
from src.data import InputHandler, OutputHandler, LtQuickDraw
from src.lightning_models import LtSketchReconstruction

import warnings
warnings.filterwarnings("ignore", message=".*tensorboardX.*")

parser = argparse.ArgumentParser()
#training arguments
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden_dropout", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=75000)
parser.add_argument("--val_check_interval", type=int, default=1500)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--log_every_n_steps", type=int, default=500)

#architecture arguments
parser.add_argument("--num_encoder_layers", type=int, default=4)
parser.add_argument("--num_decoder_layers", type=int, default=4)
parser.add_argument("--decoder_type", type=str, default="ar")
parser.add_argument("--hidden_dim", type=int, default=128)

#positional embedding arguments
parser.add_argument("--input_relative_coords", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--output_relative_coords", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--pen_state", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stroke_embedding", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--sketch_pos", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stroke_pos", action=argparse.BooleanOptionalAction, default=False)

#general arguments
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--results_path", type=str)

args = parser.parse_args()

seed_everything(args.seed, workers=True)

sketchformer = Sketchformer(
    hidden_dim=args.hidden_dim,
    num_encoder_layers=args.num_encoder_layers,
    num_decoder_layers=args.num_decoder_layers,
    decoder_type=args.decoder_type,
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



input_handler = InputHandler(input_relative_coords=args.input_relative_coords, output_relative_coords=args.output_relative_coords, autoencoder=True, autoregressive=args.decoder_type in ["ar", "ar-enc"])
output_handler = OutputHandler(output_relative_coords=args.output_relative_coords, autoregressive=args.decoder_type in ["ar", "ar-enc"])
model = LtSketchReconstruction(sketchformer, input_handler, output_handler, args.lr)
datamodule = LtQuickDraw(dataset_path="./data/quickdraw/",
                        loader_args={"seed": args.seed,
                                    "num_workers": args.num_workers,
                                    "batch_size": args.batch_size})



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
    max_steps=args.max_steps,
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir=args.ckpt_path,
    log_every_n_steps=args.log_every_n_steps,
    val_check_interval=args.val_check_interval,
    check_val_every_n_epoch=None,
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