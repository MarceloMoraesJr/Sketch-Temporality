{
    python3 -m experiments.reconstruction --base_config ./configs/decoder_studies/modifications/base_config.yaml --config ./configs/decoder_studies/modifications/mod_dae_3_seed_42.yaml
    python3 -m experiments.reconstruction --base_config ./configs/decoder_studies/modifications/base_config.yaml --config ./configs/decoder_studies/modifications/mod_dae_4_seed_42.yaml

    # for config_file in ./configs/decoder_studies/ffn/*.yaml
    # do
    #     python3 -m experiments.reconstruction --base_config ./configs/decoder_studies/modifications/base_config.yaml --config "$config_file"
    # done

    exit
}
