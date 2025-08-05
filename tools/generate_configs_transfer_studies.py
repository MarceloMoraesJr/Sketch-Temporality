# parser = argparse.ArgumentParser()
# #training arguments
# parser.add_argument("--split", type=int, default=0)
# parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--device", type=int, default=0)
# parser.add_argument("--batch_size", type=int, default=256)
# parser.add_argument("--lr", type=float, default=1e-3)
# parser.add_argument("--hidden_dropout", type=float, default=0.1)
# parser.add_argument("--num_workers", type=int, default=8)
# parser.add_argument("--max_epochs", type=int, default=150)
# parser.add_argument("--patience", type=int, default=15)

# #architecture arguments
# parser.add_argument("--num_layers", type=int, default=4)
# parser.add_argument("--hidden_dim", type=int, default=128)

# #positional embedding arguments
# parser.add_argument("--relative_coords", action=argparse.BooleanOptionalAction, default=True)
# parser.add_argument("--pen_state", action=argparse.BooleanOptionalAction, default=True)
# parser.add_argument("--stroke_embedding", action=argparse.BooleanOptionalAction, default=False)
# parser.add_argument("--sketch_pos", action=argparse.BooleanOptionalAction, default=True)
# parser.add_argument("--stroke_pos", action=argparse.BooleanOptionalAction, default=False)

# #perturbation arguments
# parser.add_argument("--inter_stroke", action=argparse.BooleanOptionalAction, default=False)
# parser.add_argument("--intra_stroke", action=argparse.BooleanOptionalAction, default=False)
# parser.add_argument("--intra_stroke_rev", action=argparse.BooleanOptionalAction, default=False)
# parser.add_argument("--stroke_order", action=argparse.BooleanOptionalAction, default=False)

# #general arguments
# parser.add_argument("--ckpt_path", type=str)
# parser.add_argument("--results_path", type=str)

# args = parser.parse_args()