{
    # PIKACHU
    # for seed in 42 1999 5342
    # do 
    #     python3 -m experiments.experiment1 --device 1 --seed ${seed} --relative_coords --pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run1_${seed}_relative/ --results_path ./results/test4/run1_${seed}_relative.pkl
    #     python3 -m experiments.experiment1 --device 1 --seed ${seed} --no-relative_coords --pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run1_${seed}_absolute/ --results_path ./results/test4/run1_${seed}_absolute.pkl
    # done
    # for seed in 42 1999 5342
    # do 
    #     python3 -m experiments.experiment1 --device 1 --seed ${seed} --relative_coords --no-pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run3_${seed}_relative/ --results_path ./results/test4/run3_${seed}_relative.pkl
    #     python3 -m experiments.experiment1 --device 1 --seed ${seed} --no-relative_coords --no-pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run3_${seed}_absolute/ --results_path ./results/test4/run3_${seed}_absolute.pkl
    # done
    # for seed in 42 1999 5342
    # do 
    #     python3 -m experiments.experiment1 --device 1 --seed ${seed} --relative_coords --no-pen_state --no-stroke_embedding --no-sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run5_${seed}_relative/ --results_path ./results/test4/run5_${seed}_relative.pkl
    #     python3 -m experiments.experiment1 --device 1 --seed ${seed} --no-relative_coords --no-pen_state --no-stroke_embedding --no-sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run5_${seed}_absolute/ --results_path ./results/test4/run5_${seed}_absolute.pkl
    # done

    # for seed in 42 1999 5342
    # do 
    #     python3 -m experiments.experiment1 --device 1 --seed ${seed} --relative_coords --pen_state --no-stroke_embedding --no-sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run6_${seed}_relative/ --results_path ./results/test4/run6_${seed}_relative.pkl
    #     python3 -m experiments.experiment1 --device 1 --seed ${seed} --no-relative_coords --pen_state --no-stroke_embedding --no-sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run6_${seed}_absolute/ --results_path ./results/test4/run6_${seed}_absolute.pkl
    # done

    # for seed in 42 1999 5342
    # do 
    #     python3 -m experiments.experiment1 --device 1 --seed ${seed} --relative_coords --no-pen_state --stroke_embedding --no-sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run7_${seed}_relative/ --results_path ./results/test4/run7_${seed}_relative.pkl
    #     python3 -m experiments.experiment1 --device 1 --seed ${seed} --no-relative_coords --no-pen_state --stroke_embedding --no-sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run7_${seed}_absolute/ --results_path ./results/test4/run7_${seed}_absolute.pkl
    # done

    # # CHARMANDER
    # for seed in 42 1999 5342
    # do 
    #     python3 -m experiments.experiment1 --device 0 --seed ${seed} --relative_coords --no-pen_state --stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/test4/run2_${seed}_relative/ --results_path ./results/test4/run2_${seed}_relative.pkl
    #     python3 -m experiments.experiment1 --device 0 --seed ${seed} --no-relative_coords --no-pen_state --stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/test4/run2_${seed}_absolute/ --results_path ./results/test4/run2_${seed}_absolute.pkl
    # done
    # for seed in 42 1999 5342
    # do 
    #     python3 -m experiments.experiment1 --device 0 --seed ${seed} --relative_coords --no-pen_state --stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run4_${seed}_relative/ --results_path ./results/test4/run4_${seed}_relative.pkl
    #     python3 -m experiments.experiment1 --device 0 --seed ${seed} --no-relative_coords --no-pen_state --stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test4/run4_${seed}_absolute/ --results_path ./results/test4/run4_${seed}_absolute.pkl
    # done

    for seed in 42 1999 5342
    do 
        python3 -m experiments.experiment1 --device 0 --seed ${seed} --relative_coords --no-pen_state --no-stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/test4/run8_${seed}_relative/ --results_path ./results/test4/run8_${seed}_relative.pkl
        python3 -m experiments.experiment1 --device 0 --seed ${seed} --no-relative_coords --no-pen_state --no-stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/test4/run8_${seed}_absolute/ --results_path ./results/test4/run8_${seed}_absolute.pkl
    done

    for seed in 42 1999 5342
    do 
        python3 -m experiments.experiment1 --device 0 --seed ${seed} --relative_coords --pen_state --no-stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/test4/run9_${seed}_relative/ --results_path ./results/test4/run9_${seed}_relative.pkl
        python3 -m experiments.experiment1 --device 0 --seed ${seed} --no-relative_coords --pen_state --no-stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/test4/run9_${seed}_absolute/ --results_path ./results/test4/run9_${seed}_absolute.pkl
    done

    exit
}