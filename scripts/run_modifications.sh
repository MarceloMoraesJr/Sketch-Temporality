{
    for (( i=2; i<5; i++ ))
    do
        python3 -m experiments.reconstruction --base_config ./configs/decoder_studies/modifications/base_config.yaml --config ./configs/decoder_studies/modifications/mod_${i}_seed_42.yaml 
    done

    exit
}
