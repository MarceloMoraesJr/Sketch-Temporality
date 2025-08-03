{
    # CHARMANDER
    for decoder_type in ar ar-enc
    do
        for seed in 42 1999 5342
        do 
            for coord_system in rel_rel rel_abs abs_rel abs_abs
            do
                python3 -m experiments.experiment4b --device 0 \
                    --input_ckpt_path ../sketch_representations/checkpoints/experiment2/${coord_system}_${decoder_type}_${seed}/best.ckpt \
                    --input_results_path ../sketch_representations/results/experiment2/${coord_system}_${decoder_type}_${seed}.pkl \
                    --output_ckpt_path ./checkpoints/experiment4b/${coord_system}_${decoder_type}_${seed} \
                    --output_results_path ./results/experiment4b/${coord_system}_${decoder_type}_${seed}.pkl
            done
        done
    done

    exit
}