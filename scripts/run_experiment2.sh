{
    # PIKACHU 
    # for decoder_type in nar nar-enc ffn
    # do
    #     for seed in 42 1999 5342
    #     do 
    #         python3 -m experiments.experiment2 --device 1 --seed ${seed} --decoder_type ${decoder_type} --input_relative_coords --output_relative_coords --ckpt_path ./checkpoints/experiment2/rel_rel_${decoder_type}_${seed} --results_path ./results/experiment2/rel_rel_${decoder_type}_${seed}.pkl
    #         python3 -m experiments.experiment2 --device 1 --seed ${seed} --decoder_type ${decoder_type} --input_relative_coords --no-output_relative_coords --ckpt_path ./checkpoints/experiment2/rel_abs_${decoder_type}_${seed} --results_path ./results/experiment2/rel_abs_${decoder_type}_${seed}.pkl
    #         python3 -m experiments.experiment2 --device 1 --seed ${seed} --decoder_type ${decoder_type} --no-input_relative_coords --output_relative_coords --ckpt_path ./checkpoints/experiment2/abs_rel_${decoder_type}_${seed} --results_path ./results/experiment2/abs_rel_${decoder_type}_${seed}.pkl
    #         python3 -m experiments.experiment2 --device 1 --seed ${seed} --decoder_type ${decoder_type} --no-input_relative_coords --no-output_relative_coords --ckpt_path ./checkpoints/experiment2/abs_abs_${decoder_type}_${seed} --results_path ./results/experiment2/abs_abs_${decoder_type}_${seed}.pkl
    #     done
    # done

    # PIKACHU 
    for decoder_type in ar
    do
        for seed in 42 1999 5342
        do 
            python3 -m experiments.experiment2 --device 0 --seed ${seed} --decoder_type ${decoder_type} --input_relative_coords --output_relative_coords --ckpt_path ./checkpoints/experiment2/rel_rel_${decoder_type}_${seed} --results_path ./results/experiment2/rel_rel_${decoder_type}_${seed}.pkl
            python3 -m experiments.experiment2 --device 0 --seed ${seed} --decoder_type ${decoder_type} --input_relative_coords --no-output_relative_coords --ckpt_path ./checkpoints/experiment2/rel_abs_${decoder_type}_${seed} --results_path ./results/experiment2/rel_abs_${decoder_type}_${seed}.pkl
            python3 -m experiments.experiment2 --device 0 --seed ${seed} --decoder_type ${decoder_type} --no-input_relative_coords --output_relative_coords --ckpt_path ./checkpoints/experiment2/abs_rel_${decoder_type}_${seed} --results_path ./results/experiment2/abs_rel_${decoder_type}_${seed}.pkl
            python3 -m experiments.experiment2 --device 0 --seed ${seed} --decoder_type ${decoder_type} --no-input_relative_coords --no-output_relative_coords --ckpt_path ./checkpoints/experiment2/abs_abs_${decoder_type}_${seed} --results_path ./results/experiment2/abs_abs_${decoder_type}_${seed}.pkl
        done
    done

    exit
}