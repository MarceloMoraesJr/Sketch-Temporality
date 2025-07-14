{   
    coord_lookup=("abs")

    i=0
    for coord in --no-relative_coords
    do
        split=0
        for seed in 42 1999 2017 5342 144256
        do
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} --no-pen_state --no-stroke_embedding --no-sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/experiment3/${coord_lookup[i]}_pe1_${split}/ --results_path ./results/experiment3/${coord_lookup[i]}_pe1_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} --pen_state --no-stroke_embedding --no-sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/experiment3/${coord_lookup[i]}_pe2_${split}/ --results_path ./results/experiment3/${coord_lookup[i]}_pe2_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} --no-pen_state --stroke_embedding --no-sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/experiment3/${coord_lookup[i]}_pe3_${split}/ --results_path ./results/experiment3/${coord_lookup[i]}_pe3_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} --no-pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/experiment3/${coord_lookup[i]}_pe4_${split}/ --results_path ./results/experiment3/${coord_lookup[i]}_pe4_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} --no-pen_state --no-stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/experiment3/${coord_lookup[i]}_pe5_${split}/ --results_path ./results/experiment3/${coord_lookup[i]}_pe5_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} --pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/experiment3/${coord_lookup[i]}_pe6_${split}/ --results_path ./results/experiment3/${coord_lookup[i]}_pe6_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} --pen_state --no-stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/experiment3/${coord_lookup[i]}_pe7_${split}/ --results_path ./results/experiment3/${coord_lookup[i]}_pe7_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} --no-pen_state --stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/experiment3/${coord_lookup[i]}_pe8_${split}/ --results_path ./results/experiment3/${coord_lookup[i]}_pe8_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} --no-pen_state --stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/experiment3/${coord_lookup[i]}_pe9_${split}/ --results_path ./results/experiment3/${coord_lookup[i]}_pe9_${split}.pkl
            ((split++))
        done
        ((i++))
    done
    exit
}