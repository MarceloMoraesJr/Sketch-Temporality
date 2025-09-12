import yaml
from pathlib import Path

config_path = Path("./configs/transfer/denoising")
config_path.mkdir(parents=True, exist_ok=True)


for i in range(5):
    config = {'reference': False, 'paths': {}}
    
    for seed in [42, 1999, 5342]:

        filename = f"decoder_{i}_seed_{seed}"
        path = config_path.joinpath(f"{filename}.yaml")
        
        config['paths']['model_base_config'] = f"./configs/decoder_studies/denoising/base_config.yaml"
        config['paths']['model_config'] = f"./configs/decoder_studies/denoising/{filename}.yaml"
        config['paths']['checkpoint'] = f"./checkpoints/transfer/denoising/{filename}"
        config['paths']['results'] = f"./results/transfer/denoising/{filename}.pkl"

        with open(path, "w") as file:
            yaml.safe_dump(config, file)