import yaml
from pathlib import Path
from sklearn.model_selection import ParameterGrid

base_config = {
    "training":{
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_dropout": 0.1,
        "num_workers": 8,
        "max_steps": 150000,
        "val_check_interval": 1500,
        "patience": 15,
        "log_every_n_steps": 500,
        "seed": 42
    },
    "architecture":{
        "num_layers": 4,
        "hidden_dim": 128
    },
    "perturbations":{
        "inter_stroke": False,
        "intra_stroke": False,
        "intra_stroke_rev": False,
        "stroke_order": False
    },
    "pe":{
        "input_relative_coords": True,
        "pen_state": True,
        "stroke_embedding": False,
        "sketch_pos": True,
        "stroke_pos": False
    }
}

base_config_path = Path("./configs/pe/classification/base_config.yaml")
base_config_path.parent.mkdir(parents=True, exist_ok=True)
with open(base_config_path, "w") as file:
    yaml.safe_dump(base_config, file)

grid = {
    "pen_state": [False, True],
    "stroke_embedding": [False, True],
    "sketch_pos": [False, True],
    "stroke_pos": [False, True]
}

pes = list(ParameterGrid(grid))

config_path = Path("./configs/pe/classification/")

for normalization in [("rel", True), ("abs", False)]:
    i = 0
    for pe in pes:
        config = {}
        config = {'training': {}, 'paths': {}}
        config['pe'] = pe
        config['pe']['input_relative_coords'] = normalization[1]
        
        if pe['pen_state'] and pe['stroke_embedding']: continue
        if pe['stroke_pos'] and pe['sketch_pos']: continue
        
        for seed in [42, 1999, 5342]:
            config['training']['seed'] = seed

            filename = f"{normalization[0]}_pe_{i}_seed_{seed}"
            path = config_path.joinpath(f"{filename}.yaml")
            
            config['paths']['checkpoint'] = f"./checkpoints/pe/classification/{filename}"
            config['paths']['results'] = f"./results/pe/classification/{filename}.pkl"

            with open(path, "w") as file:
                yaml.safe_dump(config, file)

        i += 1

base_config = {
    "training":{
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_dropout": 0.1,
        "num_workers": 8,
        "max_epochs": 150,
        "patience": 15,
        "split": 0,
        "seed": 42
    },
    "architecture":{
        "num_layers": 4,
        "hidden_dim": 128
    },
    "perturbations":{
        "inter_stroke": False,
        "intra_stroke": False,
        "intra_stroke_rev": False,
        "stroke_order": False
    },
    "pe":{
        "input_relative_coords": True,
        "pen_state": True,
        "stroke_embedding": False,
        "sketch_pos": True,
        "stroke_pos": False
    }
}

base_config_path = Path("./configs/pe/segmentation/base_config.yaml")
base_config_path.parent.mkdir(parents=True, exist_ok=True)
with open(base_config_path, "w") as file:
    yaml.safe_dump(base_config, file)

config_path = Path("./configs/pe/segmentation/")

for normalization in [("rel", True), ("abs", False)]:
    i = 0
    for pe in pes:
        config = {}
        config = {'training': {}, 'paths': {}}
        config['pe'] = pe
        config['pe']['input_relative_coords'] = normalization[1]
        
        if pe['pen_state'] and pe['stroke_embedding']: continue
        if pe['stroke_pos'] and pe['sketch_pos']: continue
        
        for split, seed in enumerate([42, 1999, 2017, 5342, 144256]):
            config['training']['seed'] = seed
            config['training']['split'] = split

            filename = f"{normalization[0]}_pe_{i}_split_{split}"
            path = config_path.joinpath(f"{filename}.yaml")
            
            config['paths']['checkpoint'] = f"./checkpoints/pe/segmentation/{filename}"
            config['paths']['results'] = f"./results/pe/segmentation/{filename}.pkl"

            with open(path, "w") as file:
                yaml.safe_dump(config, file)

        i += 1



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

base_config_path = Path("./configs/pe/reconstruction/base_config.yaml")
base_config_path.parent.mkdir(parents=True, exist_ok=True)
with open(base_config_path, "w") as file:
    yaml.safe_dump(base_config, file)

config_path = Path("./configs/pe/reconstruction/")

for normalization in [("rel", True), ("abs", False)]:
    i = -1
    for pe in pes:
        config = {}
        config = {'training': {}, 'paths': {}}
        config['pe'] = pe
        config['pe']['input_relative_coords'] = normalization[1]
        
        if pe['pen_state'] and pe['stroke_embedding']: continue
        if pe['stroke_pos'] and pe['sketch_pos']: continue
        i += 1
        if pe['pen_state'] and pe['sketch_pos']: continue # already done
        
        for seed in [42, 1999, 5342]:
            config['training']['seed'] = seed

            filename = f"{normalization[0]}_pe_{i}_seed_{seed}"
            path = config_path.joinpath(f"{filename}.yaml")
            
            config['paths']['checkpoint'] = f"./checkpoints/pe/reconstruction/{filename}"
            config['paths']['results'] = f"./results/pe/reconstruction/{filename}.pkl"

            with open(path, "w") as file:
                yaml.safe_dump(config, file)
