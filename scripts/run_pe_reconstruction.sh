{
    for normalization in abs
    do
        for ((i=4; i<=7; i++)) 
        do
            for seed in 42 1999 5342
            do
                python3 -m experiments.reconstruction --base_config ./configs/pe/reconstruction/base_config.yaml --config ./configs/pe/reconstruction/${normalization}_pe_${i}_seed_${seed}.yaml --device 1
            done
        done
    done
    exit
}