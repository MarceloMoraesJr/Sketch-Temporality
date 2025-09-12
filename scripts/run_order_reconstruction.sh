{
    python3 -m experiments.reconstruction --base_config ./configs/decoder_studies/denoising/base_config.yaml --config ./configs/decoder_studies/denoising/test_seed_42.yaml --device 0

    for normalization in rel abs
    do
        for ((i=3; i<=5; i++)) 
        do
            for seed in 42 1999 5342
            do
                python3 -m experiments.reconstruction --base_config ./configs/order/reconstruction/base_config.yaml --config ./configs/order/reconstruction/${normalization}_order_${i}_seed_${seed}.yaml --device 0
            done
        done
    done
    exit
}