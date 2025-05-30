import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNLayer(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward, dropout = 0.1, activation = F.relu):
        super(FFNLayer, self).__init__()

        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.lin1 = nn.Linear(hidden_dim, dim_feedforward)
        self.lin2 = nn.Linear(dim_feedforward, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.ln(x)
        
        h = self.lin1(h)
        h = self.activation(h)
        h = self.dropout1(h)

        h = self.lin2(h)
        h = self.dropout2(h)

        h = h + x

        return h