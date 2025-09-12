import yaml
from pathlib import Path

config_path = Path("./configs/transfer/ffn/")
config_path.mkdir(parents=True, exist_ok=True)


for input_normalization in [("rel", True), ("abs", False)]:
    for output_normalization in [("rel", True), ("abs", False)]:
        i = 0
        
        config = {'reference': False, 'paths': {}}
        
        for seed in [42, 1999, 5342]:

            filename = f"{input_normalization[0]}_{output_normalization[0]}_decoder_ffn_seed_{seed}"
            path = config_path.joinpath(f"{filename}.yaml")
            
            config['paths']['model_base_config'] = f"./configs/decoder_studies/modifications/base_config.yaml"
            config['paths']['model_config'] = f"./configs/decoder_studies/ffn/{filename}.yaml"
            config['paths']['checkpoint'] = f"./checkpoints/transfer/decoder_studies/ffn/{filename}"
            config['paths']['results'] = f"./results/transfer/decoder_studies/ffn/{filename}.pkl"

            with open(path, "w") as file:
                yaml.safe_dump(config, file)

        i += 1
