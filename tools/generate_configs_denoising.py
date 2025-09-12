import yaml
from pathlib import Path

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
            "autoregressive": True,
            "cross_attn": True,
            "condition_first": False,
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
    "denoising": True,
    "noise_std": 0.1,
    "mask_rate": 0.5
}

base_config_path = Path("./configs/decoder_studies/denoising/base_config.yaml")
base_config_path.parent.mkdir(parents=True, exist_ok=True)
with open(base_config_path, "w") as file:
    yaml.safe_dump(base_config, file)

decoders = [
    {
        "autoregressive": False,
        "cross_attn": False,
        "condition_first": True,
        "remove_self_attn": False
    },
    {
        "autoregressive": False,
        "cross_attn": False,
        "condition_first": True,
        "remove_self_attn": True,
        "replace_self_attn": True
    },
    {
        "autoregressive": False,
        "cross_attn": True,
        "condition_first": False,
        "remove_self_attn": False
    },
    {
        "autoregressive": True,
        "cross_attn": False,
        "condition_first": True,
        "remove_self_attn": False
    },
    {
        "autoregressive": True,
        "cross_attn": True,
        "condition_first": False,
        "remove_self_attn": False
    }
]

config_path = Path("./configs/decoder_studies/denoising/")

i = 0
for decoder in decoders:
    config = {'training': {}, 'paths': {}, 'architecture': {}}
    config['architecture']['decoder'] = decoder
    
    for seed in [42, 1999, 5342]:
        config['training']['seed'] = seed

        filename = f"decoder_{i}_seed_{seed}"
        path = config_path.joinpath(f"{filename}.yaml")
        
        config['paths']['checkpoint'] = f"./checkpoints/decoder_studies/denoising/{filename}"
        config['paths']['results'] = f"./results/decoder_studies/denoising/{filename}.pkl"

        with open(path, "w") as file:
            yaml.safe_dump(config, file)

    i += 1

