{
    # PIKACHU 
    python3 -m experiments.experiment1 --device 1 --lr 1e-5 --hidden_dropout 0.3 --relative_coords --pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test3/run1_relative/ --results_path ./results/test3/run1_relative.pkl
    python3 -m experiments.experiment1 --device 1 --lr 1e-5 --hidden_dropout 0.3 --no-relative_coords --pen_state --no-stroke_embedding --sketch_pos --no-stroke_pos --ckpt_path ./checkpoints/test3/run1_absolute/ --results_path ./results/test3/run1_absolute.pkl

    # CHARMANDER
    # python3 -m experiments.experiment1 --device 0 --lr 1e-5 --hidden_dropout 0.3 --relative_coords --no-pen_state --stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/test3/run2_relative/ --results_path ./results/test3/run2_relative.pkl
    # python3 -m experiments.experiment1 --device 0 --lr 1e-5 --hidden_dropout 0.3 --no-relative_coords --no-pen_state --stroke_embedding --no-sketch_pos --stroke_pos --ckpt_path ./checkpoints/test3/run2_absolute/ --results_path ./results/test3/run2_absolute.pkl

    exit
}