import torch
import torch.nn as nn
from .sketchformer import BlockConfig, TokenEmbeddingConfig, PosEmbeddingConfig, Sketchformer

class SketchformerSeg(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 num_classes: int = 10,
                 block_config: BlockConfig = BlockConfig(),
                 token_embedding_config: TokenEmbeddingConfig = TokenEmbeddingConfig(),
                 pos_embedding_config: PosEmbeddingConfig = PosEmbeddingConfig(),
                ):        
        super(SketchformerSeg, self).__init__()

        self.sketchformer = Sketchformer(
            hidden_dim=hidden_dim,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            decoder_type=None,
            block_config=block_config,
            token_embedding_config=token_embedding_config,
            pos_embedding_config=pos_embedding_config
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, pos, pos_info, token_ids, mask=None):
        point_embeddings = self.sketchformer.encode(pos, pos_info, token_ids, mask, pool=False)
        logits = self.classifier(point_embeddings)
        return logits
