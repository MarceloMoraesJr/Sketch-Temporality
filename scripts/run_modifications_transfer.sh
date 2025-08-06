{
    for (( i=2; i<4; i++ ))
    do
        python3 -m experiments.transfer_fb --base_config ./configs/transfer/modifications/base_config.yaml --config ./configs/transfer/modifications/mod_${i}_seed_42.yaml 
    done

    exit
}
