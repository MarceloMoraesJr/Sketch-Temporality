{   
    coord_lookup=("rel" "abs")
    pe_lookup=("--pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos" "--pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos")

    i=0
    for coord in --relative_coords --no-relative_coords
    do
        for seed in 42 1999 5342
        do
            python3 -m experiments.experiment1 --device 1 --seed ${seed} ${coord} ${pe_lookup[i]} --inter_stroke --no-intra_stroke --no-intra_stroke_rev --no-stroke_order --ckpt_path ./checkpoints/experiment1_order/${coord_lookup[i]}_order1_${seed}/ --results_path ./results/experiment1_order/${coord_lookup[i]}_order1_${seed}.pkl
            python3 -m experiments.experiment1 --device 1 --seed ${seed} ${coord} ${pe_lookup[i]} --no-inter_stroke --intra_stroke --no-intra_stroke_rev --no-stroke_order --ckpt_path ./checkpoints/experiment1_order/${coord_lookup[i]}_order2_${seed}/ --results_path ./results/experiment1_order/${coord_lookup[i]}_order2_${seed}.pkl
            python3 -m experiments.experiment1 --device 1 --seed ${seed} ${coord} ${pe_lookup[i]} --no-inter_stroke --no-intra_stroke --intra_stroke_rev --no-stroke_order --ckpt_path ./checkpoints/experiment1_order/${coord_lookup[i]}_order3_${seed}/ --results_path ./results/experiment1_order/${coord_lookup[i]}_order3_${seed}.pkl
            python3 -m experiments.experiment1 --device 1 --seed ${seed} ${coord} ${pe_lookup[i]} --no-inter_stroke --no-intra_stroke --no-intra_stroke_rev --stroke_order --ckpt_path ./checkpoints/experiment1_order/${coord_lookup[i]}_order4_${seed}/ --results_path ./results/experiment1_order/${coord_lookup[i]}_order4_${seed}.pkl
            python3 -m experiments.experiment1 --device 1 --seed ${seed} ${coord} ${pe_lookup[i]} --no-inter_stroke --intra_stroke --no-intra_stroke_rev --stroke_order --ckpt_path ./checkpoints/experiment1_order/${coord_lookup[i]}_order5_${seed}/ --results_path ./results/experiment1_order/${coord_lookup[i]}_order5_${seed}.pkl
            python3 -m experiments.experiment1 --device 1 --seed ${seed} ${coord} ${pe_lookup[i]} --no-inter_stroke --no-intra_stroke --intra_stroke_rev --stroke_order --ckpt_path ./checkpoints/experiment1_order/${coord_lookup[i]}_order6_${seed}/ --results_path ./results/experiment1_order/${coord_lookup[i]}_order6_${seed}.pkl
        done
        ((i++))
    done
    exit
}