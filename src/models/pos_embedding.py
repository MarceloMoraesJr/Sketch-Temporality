import torch
import torch.nn as nn
import math
from dataclasses import dataclass

@dataclass
class PosEmbeddingConfig():
    pen_state: bool = True
    stroke_embedding: bool = False
    sketch_pos_pe: bool = True
    stroke_pos_pe: bool = False
    max_strokes: int = 1000 
    max_len_sin_pe: int = 5000 



class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPosEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, position=None):
        if position is None:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, position, :].squeeze(0)
        return x
    

    
class PosEmbedding(nn.Module):
    def __init__(self, hidden_dim, pen_state=True, stroke_embedding=False, sketch_pos_pe=True, stroke_pos_pe=False, max_strokes=1000, max_len_sin_pe=5000):
        super(PosEmbedding, self).__init__()

        assert not sketch_pos_pe or not stroke_pos_pe

        if pen_state:
            self.pen_state_proj = nn.Linear(3, hidden_dim)
        
        if stroke_embedding:
            self.stroke_embedding = nn.Embedding(max_strokes, hidden_dim)

        if sketch_pos_pe or stroke_pos_pe:
            self.pos_pe = SinusoidalPosEmbedding(hidden_dim, max_len_sin_pe)
        self.stroke_pos_pe = stroke_pos_pe


    def forward(self, x, pos_info):
        if hasattr(self, "pen_state_proj"):
            x = x + self.pen_state_proj(pos_info['pen_state'])

        if hasattr(self, "stroke_embedding"):
            x = x + self.stroke_embedding(pos_info['stroke_id'])
             
        if hasattr(self, "pos_pe"):
            x = x + self.pos_pe(x, pos_info['stroke_pos'] if self.stroke_pos_pe else None)

        return x