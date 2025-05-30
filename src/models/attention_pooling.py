import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super(AttentionPooling, self).__init__()

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        s = self.lin1(x).tanh()
        s = self.lin2(s)

        if mask is not None:
            s = s.masked_fill(mask.unsqueeze(dim=-1), -torch.inf)

        s = nn.Softmax(dim = 1)(s)
        return (s * x).sum(dim=1)