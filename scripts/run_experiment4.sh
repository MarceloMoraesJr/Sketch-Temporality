{
    # CHARMANDER

    #FINE-TUNING
    for decoder_type in ar ar-enc
    do
        for seed in 42 1999 5342
        do 
            for coord_system in rel_rel rel_abs abs_rel abs_abs
            do
                python3 -m experiments.experiment4 --device 0 \
                    --input_ckpt_path ../sketch_representations/checkpoints/experiment2/${coord_system}_${decoder_type}_${seed}/best.ckpt \
                    --input_results_path ../sketch_representations/results/experiment2/${coord_system}_${decoder_type}_${seed}.pkl \
                    --output_ckpt_path ./checkpoints/experiment4/${coord_system}_${decoder_type}_${seed}.ckpt \
                    --output_results_path ./results/experiment4/${coord_system}_${decoder_type}_${seed}.pkl
            done
        done
    done

    #REFERENCE (SCRATCH)
    # for seed in 42 1999 5342
    # do
    #     python3 -m experiments.experiment4 --device 0 \
    #         --reference --seed ${seed} \
    #         --input_relative_coords \
    #         --output_ckpt_path ./checkpoints/experiment4/reference_rel_${seed}.ckpt \
    #         --output_results_path ./results/experiment4/reference_rel_${seed}.pkl

    #     python3 -m experiments.experiment4 --device 0 \
    #         --reference --seed ${seed} \
    #         --no-input_relative_coords \
    #         --output_ckpt_path ./checkpoints/experiment4/reference_abs_${seed}.ckpt \
    #         --output_results_path ./results/experiment4/reference_abs_${seed}.pkl
    # done

    exit
}