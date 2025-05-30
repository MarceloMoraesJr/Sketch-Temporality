import argparse
import pickle as pkl

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models import SketchformerClassifier, BlockConfig, PosEmbeddingConfig
from src.data import InputHandler, LtQuickDraw
from src.lightning_models import LtSketchClassification

import warnings
warnings.filterwarnings("ignore", message=".*tensorboardX.*")

parser = argparse.ArgumentParser()
#training arguments
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--max_epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--hidden_dropout", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=8)

#architecture arguments
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--hidden_dim", type=int, default=128)

#positional embedding arguments
parser.add_argument("--relative_coords", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--pen_state", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stroke_embedding", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--sketch_pos_pe", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stroke_pos_pe", action=argparse.BooleanOptionalAction, default=False)

#general arguments
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--results_path", type=str, default=None)

args = parser.parse_args()

seed_everything(args.seed, workers=True)

sketchformer = SketchformerClassifier(
    hidden_dim=args.hidden_dim,
    num_layers=args.num_layers,
    num_classes=345,
    block_config=BlockConfig(
        dropout=args.hidden_dropout
    ),
    pos_embedding_config=PosEmbeddingConfig(
        pen_state=args.pen_state, 
        stroke_embedding=args.stroke_embedding, 
        sketch_pos_pe=args.sketch_pos_pe,
        stroke_pos_pe=args.stroke_pos_pe
    )
)



input_handler = InputHandler()
model = LtSketchClassification(sketchformer, input_handler)
datamodule = LtQuickDraw(dataset_path="./data/quickdraw/",
                        dataset_args={"relative_coords": args.relative_coords},
                        loader_args={"seed": args.seed,
                                    "num_workers": args.num_workers,
                                    "batch_size": args.batch_size})



checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=args.ckpt_path,
    mode="min",
    save_top_k=1,
    filename="best",
    verbose=args.verbose
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=args.patience,
    mode="min"
)

trainer = Trainer(
    max_epochs=args.max_epochs,
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir=args.ckpt_path,
    log_every_n_steps=50,
    val_check_interval=0.25,
    deterministic=args.verbose,
    accelerator="gpu",
    devices=[args.device]
)

trainer.fit(model, datamodule=datamodule)
test_acc = trainer.test(model, datamodule=datamodule, ckpt_path="best")

with open(args.result_path, "wb") as file:
    pkl.dump({"test_acc": test_acc, "args": args})