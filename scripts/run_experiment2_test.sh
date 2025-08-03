{
    # MEWTWO
    for decoder_type in nar-enc-mod
    do
        for seed in 42 1999 5342
        do 
            python3 -m experiments.experiment2 --device 0 --seed ${seed} --decoder_type ${decoder_type} --no-input_relative_coords --no-output_relative_coords --ckpt_path ./checkpoints/experiment2_test/abs_abs_${decoder_type}_${seed} --results_path ./results/experiment2_test/abs_abs_${decoder_type}_${seed}.pkl
        done
    done

    exit
}