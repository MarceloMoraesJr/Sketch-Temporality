import yaml
from pathlib import Path

config_path = Path("./configs/decoder_studies/ffn/")
config_path.mkdir(parents=True, exist_ok=True)

decoders = [
    {
        "autoregressive": False,
        "cross_attn": False,
        "condition_first": True,
        "condition_every": False,
        "remove_self_attn": True,
        "replace_self_attn": True,
        "replace_cross_attn": False
    }
]

for input_normalization in [("rel", True), ("abs", False)]:
    for output_normalization in [("rel", True), ("abs", False)]:
        i = 0
        for decoder in decoders:
            config = {'training': {}, 'paths': {}, 'pe': {}, 'architecture': {}}
            config['pe']['input_relative_coords'] = input_normalization[1]
            config['pe']['output_relative_coords'] = output_normalization[1]
            config['architecture']['decoder'] = decoder
            
            for seed in [42, 1999, 5342]:
                config['training']['seed'] = seed

                filename = f"{input_normalization[0]}_{output_normalization[0]}_decoder_ffn_seed_{seed}"
                path = config_path.joinpath(f"{filename}.yaml")
                
                config['paths']['checkpoint'] = f"./checkpoints/decoder_studies/ffn/{filename}"
                config['paths']['results'] = f"./results/decoder_studies/ffn/{filename}.pkl"

                with open(path, "w") as file:
                    yaml.safe_dump(config, file)

            i += 1
