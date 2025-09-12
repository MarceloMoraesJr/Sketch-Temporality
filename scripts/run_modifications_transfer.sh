{    
    # python3 -m experiments.transfer_ft --base_config ./configs/transfer/modifications/base_config.yaml --config ./configs/transfer/modifications/mod_dae_seed_42.yaml 
    # python3 -m experiments.transfer_ft --base_config ./configs/transfer/modifications/base_config.yaml --config ./configs/transfer/modifications/mod_dae_2_seed_42.yaml

    for file in ./configs/transfer/ffn/*
    do
        python3 -m experiments.transfer_ft --base_config ./configs/transfer/modifications/base_config.yaml --config $file --device 1
    done

    exit
}
