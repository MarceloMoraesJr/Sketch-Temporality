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
parser.add_argument("--lr_sketchformer", type=float, default=1e-3)
parser.add_argument("--lr_classifier", type=float, default=1e-3)
parser.add_argument("--num_epochs_freeze", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=2)

#general arguments
parser.add_argument("--input_ckpt_path", type=str)
parser.add_argument("--output_ckpt_path", type=str)
parser.add_argument("--input_results_path", type=str)
parser.add_argument("--output_results_path", type=str)

args = parser.parse_args()


with open(args.input_results_path, "rb") as file:
    results = pkl.load(file)

state_dict = torch.load(args.input_ckpt_path)['state_dict']

seed_everything(results['args']['seed'], workers=True)

sketchformer = SketchformerClassifier(
    hidden_dim=results['args']['hidden_dim'],
    num_layers=results['args']['num_encoder_layers'],
    num_classes=345,
    block_config=BlockConfig(
        dropout=results['args']['hidden_dropout']
    ),
    pos_embedding_config=PosEmbeddingConfig(
        pen_state=results['args']['pen_state'], 
        stroke_embedding=results['args']['stroke_embedding'], 
        sketch_pos=results['args']['sketch_pos'],
        stroke_pos=results['args']['stroke_pos']
    )
)


input_handler = InputHandler(input_relative_coords=results['args']['input_relative_coords'], autoencoder=False)
model = LtSketchClassification(sketchformer, input_handler, lr={"sketchformer": args.lr_sketchformer, "classifier": args.lr_classifier})
model.sketchformer.load_state_dict(state_dict, strict=False)

datamodule = LtQuickDraw(dataset_path="../sketch_representations/data/quickdraw/",
                        loader_args={"seed": results['args']['seed'],
                                    "num_workers": results['args']['num_workers'],
                                    "batch_size": results['args']['batch_size']})

datamodule.setup()

# # TRAINING WITH FROZEN ENCODER (JUST LINEAR LAYER)
# for param in model.sketchformer.sketchformer.parameters():
#     param.requires_grad = False
# model.sketchformer.sketchformer.eval()

trainer = Trainer(
    max_epochs=1,
    logger=False,
    enable_checkpointing=False,
    check_val_every_n_epoch=None,
    deterministic=True,
    accelerator="gpu",
    devices=[args.device]
)

#trainer.fit(model, train_dataloaders=datamodule.train_dataloader())

# # TRAINING THE WHOLE MODEL
# for param in model.sketchformer.sketchformer.parameters():
#     param.requires_grad = True
# model.sketchformer.sketchformer.train()

# trainer = Trainer(
#     max_epochs= args.max_epochs - args.num_epochs_freeze,
#     logger=False,
#     enable_checkpointing=False,
#     check_val_every_n_epoch=None,
#     deterministic=True,
#     accelerator="gpu",
#     devices=[args.device]
# )

trainer.fit(model, train_dataloaders=datamodule.train_dataloader())
trainer.test(model, dataloaders=datamodule.test_dataloader())

print(model.test_results)

# output_results_path = Path(args.output_results_path)
# output_results_path.parent.mkdir(parents=True, exist_ok=True)

# with open(output_results_path, "wb") as file:
#     pkl.dump({"test_results": model.test_results, "pretraining_args": results['args'], "finetuning_args": vars(args)}, file)