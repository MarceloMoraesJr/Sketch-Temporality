{   
    coord_lookup=("rel" "abs")
    pe_lookup=("--no-pen_state --stroke_embedding --no-sketch_pos --stroke_pos" "--pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos")

    i=0
    for coord in --relative_coords --no-relative_coords
    do
        split=0
        for seed in 42 1999 2017 5342 144256
        do
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} ${pe_lookup[i]} --inter_stroke --no-intra_stroke --no-intra_stroke_rev --no-stroke_order --ckpt_path ./checkpoints/experiment3_order/${coord_lookup[i]}_order1_${split}/ --results_path ./results/experiment3_order/${coord_lookup[i]}_order1_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} ${pe_lookup[i]} --no-inter_stroke --intra_stroke --no-intra_stroke_rev --no-stroke_order --ckpt_path ./checkpoints/experiment3_order/${coord_lookup[i]}_order2_${split}/ --results_path ./results/experiment3_order/${coord_lookup[i]}_order2_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} ${pe_lookup[i]} --no-inter_stroke --no-intra_stroke --intra_stroke_rev --no-stroke_order --ckpt_path ./checkpoints/experiment3_order/${coord_lookup[i]}_order3_${split}/ --results_path ./results/experiment3_order/${coord_lookup[i]}_order3_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} ${pe_lookup[i]} --no-inter_stroke --no-intra_stroke --no-intra_stroke_rev --stroke_order --ckpt_path ./checkpoints/experiment3_order/${coord_lookup[i]}_order4_${split}/ --results_path ./results/experiment3_order/${coord_lookup[i]}_order4_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} ${pe_lookup[i]} --no-inter_stroke --intra_stroke --no-intra_stroke_rev --stroke_order --ckpt_path ./checkpoints/experiment3_order/${coord_lookup[i]}_order5_${split}/ --results_path ./results/experiment3_order/${coord_lookup[i]}_order5_${split}.pkl
            python3 -m experiments.experiment3 --device 0 --seed ${seed} --split ${split} ${coord} ${pe_lookup[i]} --no-inter_stroke --no-intra_stroke --intra_stroke_rev --stroke_order --ckpt_path ./checkpoints/experiment3_order/${coord_lookup[i]}_order6_${split}/ --results_path ./results/experiment3_order/${coord_lookup[i]}_order6_${split}.pkl
            ((split++))
        done
        ((i++))
    done
    exit
}