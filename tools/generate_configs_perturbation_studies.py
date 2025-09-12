import yaml
from pathlib import Path
from sklearn.model_selection import ParameterGrid

base_config = {
    "training":{
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_dropout": 0.1,
        "num_workers": 8,
        "max_steps": 75000,
        "val_check_interval": 1500,
        "patience": 15,
        "log_every_n_steps": 500,
        "seed": 42
    },
    "architecture":{
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "hidden_dim": 128,
        "decoder":{
            "autoregressive": False,
            "cross_attn": False,
            "condition_first": True,
            "condition_every": False,
            "replace_cross_attn": False,
            "remove_self_attn": False,
            "replace_self_attn": False 
        }
    },
    "perturbations":{
        "inter_stroke": False,
        "intra_stroke": False,
        "intra_stroke_rev": False,
        "stroke_order": False
    },
    "pe":{
        "input_relative_coords": False,
        "output_relative_coords": False,
        "pen_state": True,
        "stroke_embedding": False,
        "sketch_pos": True,
        "stroke_pos": False
    },
    "denoising": False,
    "noise_std": 0.0,
    "mask_rate": 0.0
}

base_config_path = Path("./configs/order/reconstruction/base_config.yaml")
base_config_path.parent.mkdir(parents=True, exist_ok=True)
with open(base_config_path, "w") as file:
    yaml.safe_dump(base_config, file)
    
perturbations = [
    {"inter_stroke": True, "intra_stroke": False, "intra_stroke_rev": False, "stroke_order": False},
    {"inter_stroke": False, "intra_stroke": True, "intra_stroke_rev": False, "stroke_order": False},
    {"inter_stroke": False, "intra_stroke": False, "intra_stroke_rev": True, "stroke_order": False},
    {"inter_stroke": False, "intra_stroke": False, "intra_stroke_rev": False, "stroke_order": True},
    {"inter_stroke": False, "intra_stroke": True, "intra_stroke_rev": False, "stroke_order": True},
    {"inter_stroke": False, "intra_stroke": False, "intra_stroke_rev": True, "stroke_order": True},
]
config_path = Path("./configs/order/reconstruction/")

for normalization in [("rel", True), ("abs", False)]:
    for i, perturbation in enumerate(perturbations):
        config = {'training': {}, 'paths': {}, "pe": {}}
        config['perturbations'] = perturbation
        config['pe']['input_relative_coords'] = normalization[1]
    
        for seed in [42, 1999, 5342]:
            config['training']['seed'] = seed

            filename = f"{normalization[0]}_order_{i}_seed_{seed}"
            path = config_path.joinpath(f"{filename}.yaml")
            
            config['paths']['checkpoint'] = f"./checkpoints/order/reconstruction/{filename}"
            config['paths']['results'] = f"./results/order/reconstruction/{filename}.pkl"

            with open(path, "w") as file:
                yaml.safe_dump(config, file)
