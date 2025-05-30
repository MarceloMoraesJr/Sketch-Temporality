import argparse

from src.models import Sketchformer, BlockConfig, PosEmbeddingConfig
from src.data import InputHandler, LtQuickDraw
from src.lightning_models import LtSketchReconstruction

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import warnings
warnings.filterwarnings("ignore", message=".*tensorboardX.*")

parser = argparse.ArgumentParser()
#training arguments
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--hidden_dropout", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=4)

#architecture arguments
parser.add_argument("--num_encoder_layers", type=int, default=4)
parser.add_argument("--num_decoder_layers", type=int, default=4)
parser.add_argument("--decoder_type", type=str, default="ar")
parser.add_argument("--hidden_dim", type=int, default=128)

#model input arguments
parser.add_argument("--relative_coords", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--pen_state', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--stroke_embedding', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--sketch_pos_pe', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--stroke_pos_pe', action=argparse.BooleanOptionalAction, default=False)

#general arguments
parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)
# parser.add_argument('--log_path', type=str, default=None)
# parser.add_argument('--model_path', type=str, default=None)
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
        sketch_pos_pe=args.sketch_pos_pe,
        stroke_pos_pe=args.stroke_pos_pe
    )
).to(args.device)



input_handler = InputHandler(args.device)
model = LtSketchReconstruction(sketchformer, input_handler)
datamodule = LtQuickDraw(dataset_path="./data/quickdraw/",
                        dataset_args={"relative_coords": args.relative_coords},
                        loader_args={"seed": args.seed,
                                    "num_workers": args.num_workers,
                                    "batch_size": args.batch_size})



checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename="best-checkpoint",
    every_n_train_steps=1000,
    verbose=True
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=3,
    mode="min"
)

trainer = Trainer(
    max_epochs=20,
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir="checkpoints/experiment2",
    log_every_n_steps=50,
    val_check_interval=0.25, 
    deterministic=True,
    accelerator="gpu",
    devices=[1]
)

ckpt_path = "checkpoints/experiment2/last.ckpt"
try:
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
except FileNotFoundError:
    trainer.fit(model, datamodule=datamodule)

print(trainer.test(model, datamodule=datamodule, ckpt_path="best"))