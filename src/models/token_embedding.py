import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TokenEmbeddingConfig():
    num_special_tokens: int = 4 
    pos_token_id: int = 0
    


class TokenEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, num_special_tokens: int = 4, pos_token_id: int = 0):
        super(TokenEmbedding, self).__init__()

        self.input_proj = nn.Linear(2, hidden_dim)
        self.emb = nn.Embedding(num_special_tokens, hidden_dim, padding_idx = pos_token_id)

        self.pos_token_id = pos_token_id

    def forward(self, pos, token_id):
        input_emb = self.emb(token_id)

        if pos is not None:
            input_emb[token_id == self.pos_token_id] += self.input_proj(pos[token_id == self.pos_token_id])

        return input_emb