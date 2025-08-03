import torch
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
    def __init__(self, decoder_type: str, hidden_dim: int, num_decoder_layers: int, token_emb: TokenEmbedding, pos_emb: PosEmbedding, block_config: BlockConfig):
        super(DecoderWrapper, self).__init__()

        self.decoder_type = decoder_type
        self.token_emb = token_emb
        self.pos_emb = pos_emb
        self.ln_decoder = nn.LayerNorm(hidden_dim)


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

        elif decoder_type == "nar-enc":
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

    def forward(self, h_sketch, pos, pos_info, token_ids, mask):
        x = self.token_emb(pos, token_ids)

        # Adds PE excepts for AR at inference
        if self.decoder_type not in ["ar", "ar-enc"] or self.training:
            x = self.pos_emb(x, pos_info)
        
        # No cross-attention
        if self.decoder_type in ["nar-enc", "ffn"] or (self.decoder_type == "ar-enc" and self.training):
            x = x + h_sketch
            h_sketch = None

        # Autoregressive decoders
        if self.decoder_type in ["ar", "ar-enc"]:
            cache = None
            if self.training:
                output, _ = self.decoder(x, h_sketch, cache)
            else:
                pred = x[:, [0]] #SOS token
                for i in range(x.shape[1]):
                    step_pos_info = {k: v[:, :i+1] for k, v in pos_info.items()}
                    x = self.pos_emb(pred, step_pos_info)

                    # No cross-attention while AR inference
                    if self.decoder_type == "ar-enc":
                        x = x + h_sketch
                        output, cache = self.decoder(x, None, cache)
                        output = self.ln_decoder(output)
                    else:
                        output, cache = self.decoder(x, h_sketch, cache)
                        output = self.ln_decoder(output)

                    pred = torch.cat([pred, output[:, [-1]]], dim=1)
        # Non-autoregressive decoders
        elif self.decoder_type == "nar":
            output = self.decoder(x, h_sketch, tgt_is_causal=False, tgt_key_padding_mask=~mask)
        elif self.decoder_type == "nar-enc":
            output = self.decoder(x, src_key_padding_mask=~mask)
        elif self.decoder_type == "ffn":
            output = self.decoder(x)

        # Last layer norm except for AR at inference 
        if self.decoder_type not in ["ar", "ar-enc"] or self.training:
            output = self.ln_decoder(output)

        return output



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
        
        self.token_emb = TokenEmbedding(hidden_dim, **token_embedding_config.__dict__)
        self.pos_emb = PosEmbedding(hidden_dim, **pos_embedding_config.__dict__)

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
            self.decoder = DecoderWrapper(decoder_type, hidden_dim, num_decoder_layers, self.token_emb, self.pos_emb, block_config)
            self.out_proj = nn.Linear(hidden_dim, 2)


    def encode(self, pos, pos_info, token_ids, mask, pool=True):
        x = self.token_emb(pos, token_ids)
        x = self.pos_emb(x, pos_info)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.ln_encoder(x)

        if pool:
            x = self.pool(x, ~mask)

        return x
    
    def decode(self, h_sketch, pos, pos_info, token_ids, mask=None):
        if self.encoder_only:
            raise Exception("Encoder only model")

        h_sketch = h_sketch.unsqueeze(1)
        
        x = self.decoder(h_sketch, pos, pos_info, token_ids, mask)

        x = self.out_proj(x)
        
        return x

    def forward(self, pos, pos_info, token_ids, mask, pool=True):
        return self.encode(pos, pos_info, token_ids, mask, pool)