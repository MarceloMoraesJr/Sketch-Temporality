import torch
import numpy as np
import random

def test_reproducibility():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')

    for i in range(5):
        out = torch.matmul(x, y)
        print(f"Run {i}: sum={out.sum().item()}")

test_reproducibility()

