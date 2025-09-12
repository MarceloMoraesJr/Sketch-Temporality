{
    for ((i=0; i<=4; i++)) 
    do
        for seed in 42 1999 5342
        do
            python3 -m experiments.reconstruction --base_config ./configs/decoder_studies/denoising/base_config.yaml --config ./configs/decoder_studies/denoising/decoder_${i}_seed_${seed}.yaml --device 1
        done
    done
    exit
}