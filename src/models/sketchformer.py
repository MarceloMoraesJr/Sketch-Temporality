from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union, Callable

from .pos_embedding import PosEmbedding, PosEmbeddingConfig
from .token_embedding import TokenEmbedding, TokenEmbeddingConfig
from .causal_transformer_decoder import CausalTransformerDecoder, CausalTransformerDecoderLayer
from .ffn_layer import FFNLayer
from .attention_pooling import AttentionPooling

@dataclass
class BlockConfig():
    nhead: int = 8
    dim_feedforward: int = 512
    dropout: int = 0.1
    activation: Union[str, Callable[[Tensor], Tensor]]= F.gelu


class DecoderWrapper(nn.Module):
    def __init__(self, decoder_type: str, hidden_dim: int, num_decoder_layers: int, block_config: BlockConfig):
        super(DecoderWrapper, self).__init__()

        self.decoder_type = decoder_type

        if decoder_type in ["ar", "ar-enc"]:
            decoder_layer = CausalTransformerDecoderLayer(
                d_model=hidden_dim,
                cross_attention=decoder_type=="ar",
                norm_first=True,
                batch_first=True, 
                **block_config.__dict__
            )

            decoder = CausalTransformerDecoder(decoder_layer, num_decoder_layers)

        elif decoder_type == "nar":
            decoder_layer = nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    norm_first=True,
                    batch_first=True, 
                    **block_config.__dict__
                )

            decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        elif decoder_type in ["nar-enc"]:
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                norm_first=True,
                batch_first=True, 
                **block_config.__dict__
            )
            
            decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers, enable_nested_tensor=False)

        elif decoder_type == "ffn":
            layers = [FFNLayer(
                hidden_dim, 
                block_config.dim_feedforward, 
                block_config.dropout, 
                block_config.activation) for _ in range(num_decoder_layers)]
            decoder = nn.Sequential(*layers)
        else:
            raise ValueError("Unknown Decoder Type")
    
        self.decoder = decoder

    def forward(self, x, h_sketch, cache, mask):
        if self.decoder_type == "ar":
            output = self.decoder(x, h_sketch, cache)
        elif self.decoder_type == "ar-enc":
            x = x + h_sketch
            output = self.decoder(x, None, cache)
        elif self.decoder_type == "nar":
            output = self.decoder(x, h_sketch, tgt_is_causal=False, tgt_key_padding_mask=~mask)
        elif self.decoder_type == "nar-enc":
            x = x + h_sketch
            output = self.decoder(x, src_key_padding_mask=~mask)
        elif self.decoder_type == "ffn":
            x = x + h_sketch
            output = self.decoder(x)
        
        if isinstance(output, tuple):
            x, cache = output
        else:
            x = output

        return x, cache



class Sketchformer(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int, 
                 decoder_type: str = "ar",
                 block_config: BlockConfig = BlockConfig(),
                 token_embedding_config: TokenEmbeddingConfig = TokenEmbeddingConfig(),
                 pos_embedding_config: PosEmbeddingConfig = PosEmbeddingConfig(),
                ):        
        super(Sketchformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            norm_first=True,
            batch_first=True, 
            **block_config.__dict__
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, enable_nested_tensor=False)
        self.ln_encoder = nn.LayerNorm(hidden_dim)
        self.pool = AttentionPooling(hidden_dim)
        
        self.decoder_type = decoder_type
        self.encoder_only = decoder_type is None or num_decoder_layers == 0

        if not self.encoder_only:
            self.decoder = DecoderWrapper(decoder_type, hidden_dim, num_decoder_layers, block_config)
            self.ln_decoder = nn.LayerNorm(hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, 2)

        self.token_emb = TokenEmbedding(hidden_dim, **token_embedding_config.__dict__)
        self.pos_emb = PosEmbedding(hidden_dim, **pos_embedding_config.__dict__)

    def encode(self, pos, pos_info, token_ids, mask, pool=True):
        x = self.token_emb(pos, token_ids)
        x = self.pos_emb(x, pos_info)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.ln_encoder(x)

        if pool:
            x = self.pool(x, ~mask)

        return x
    
    def decode(self, h_sketch, pos, pos_info, token_ids, cache=None, mask=None):
        if self.encoder_only:
            raise Exception("Encoder only model")

        h_sketch = h_sketch.unsqueeze(1)

        x = self.token_emb(pos, token_ids)
        x = self.pos_emb(x, pos_info)
        
        x, cache = self.decoder(x, h_sketch, cache, mask)

        x = self.ln_decoder(x)
        x = self.out_proj(x)
        
        return x, cache

    def forward(self, pos, pos_info, mask, pool=True):
        return self.encode(pos, pos_info, mask, pool)