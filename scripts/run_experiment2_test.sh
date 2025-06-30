{
    python3 -m experiments.experiment2 --decoder_type nar-enc --input_relative_coords --output_relative_coords --ckpt_path ./checkpoints/test1/run1 --results_path ./results/test1/run1.pkl
    python3 -m experiments.experiment2 --decoder_type nar-enc --input_relative_coords --no-output_relative_coords --ckpt_path ./checkpoints/test1/run2 --results_path ./results/test1/run2.pkl
    python3 -m experiments.experiment2 --decoder_type nar-enc --no-input_relative_coords --output_relative_coords --ckpt_path ./checkpoints/test1/run3 --results_path ./results/test1/run3.pkl
    python3 -m experiments.experiment2 --decoder_type nar-enc --no-input_relative_coords --no-output_relative_coords --ckpt_path ./checkpoints/test1/run4 --results_path ./results/test1/run4.pkl

    exit
}