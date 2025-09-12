import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Union, Callable, Optional

from .pos_embedding import PosEmbedding, PosEmbeddingConfig
from .token_embedding import TokenEmbedding, TokenEmbeddingConfig
from .causal_transformer_decoder import CausalTransformerDecoder, CausalTransformerDecoderLayer
from .transformer_encoder_mod import TransformerEncoderModified, TransformerEncoderLayerModified, ModificationsConfig
from .attention_pooling import AttentionPooling

@dataclass
class BlockConfig():
    nhead: int = 8
    dim_feedforward: int = 512
    dropout: int = 0.1
    activation: Union[str, Callable[[Tensor], Tensor]]= F.gelu

@dataclass
class DecoderConfig():
    autoregressive: bool = True
    cross_attn: bool = True
    condition_first: bool = False
    modifications_config: ModificationsConfig = field(default_factory=ModificationsConfig)

    def __post_init__(self):
        assert self.cross_attn ^ (self.condition_first or self.modifications_config.condition_every) \
            or (not self.cross_attn and self.condition_first ^ self.modifications_config.condition_every)



class DecoderWrapper(nn.Module):
    def __init__(self, hidden_dim: int, num_decoder_layers: int, token_emb: TokenEmbedding, pos_emb: PosEmbedding, block_config: BlockConfig, decoder_config: DecoderConfig = DecoderConfig()):
        super(DecoderWrapper, self).__init__()

        self.decoder_config = decoder_config
        self.token_emb = token_emb
        self.pos_emb = pos_emb
        self.ln_decoder = nn.LayerNorm(hidden_dim)


        if decoder_config.autoregressive:
            decoder_layer = CausalTransformerDecoderLayer(
                d_model=hidden_dim,
                cross_attention=decoder_config.cross_attn,
                norm_first=True,
                batch_first=True, 
                **block_config.__dict__
            )

            decoder = CausalTransformerDecoder(decoder_layer, num_decoder_layers)

        elif decoder_config.cross_attn:
            decoder_layer = nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    norm_first=True,
                    batch_first=True, 
                    **block_config.__dict__
                )

            decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        else:
            decoder_layer = TransformerEncoderLayerModified(
                d_model=hidden_dim,
                norm_first=True,
                batch_first=True,
                modifications_config=decoder_config.modifications_config,
                **block_config.__dict__
            )
            
            decoder = TransformerEncoderModified(decoder_layer, num_decoder_layers, enable_nested_tensor=False)

        self.decoder = decoder

    def forward(self, h_sketch, pos, pos_info, token_ids, mask):
        x = self.token_emb(pos, token_ids)

        # Adds PE excepts for AR at inference
        if not self.decoder_config.autoregressive or self.training:
            x = self.pos_emb(x, pos_info)
        
        # Sum first only
        #if self.decoder_config.condition_first:
        if self.decoder_config.condition_first and (not self.decoder_config.autoregressive or self.training):
            x = x + h_sketch
            h_sketch = None

        # Autoregressive decoders
        if self.decoder_config.autoregressive:
            cache = None
            if self.training:
                output, _ = self.decoder(x, h_sketch, cache)
            else:
                pred = x[:, [0]] #SOS token
                for i in range(x.shape[1]):
                    step_pos_info = {k: v[:, :i+1] for k, v in pos_info.items()}
                    x = self.pos_emb(pred, step_pos_info)

                    # No cross-attention while AR inference
                    if not self.decoder_config.cross_attn:
                        x = x + h_sketch
                        output, cache = self.decoder(x, None, cache)
                        output = self.ln_decoder(output)
                    else:
                        output, cache = self.decoder(x, h_sketch, cache)
                        output = self.ln_decoder(output)

                    pred = torch.cat([pred, output[:, [-1]]], dim=1)
        # Non-autoregressive decoders
        elif self.decoder_config.cross_attn:
            output = self.decoder(x, h_sketch, tgt_is_causal=False, tgt_key_padding_mask=~mask)
        else:
            output = self.decoder(x, h_sketch, src_key_padding_mask=~mask)
    
        # Last layer norm except for AR at inference 
        if not self.decoder_config.autoregressive or self.training:
            output = self.ln_decoder(output)

        return output



class Sketchformer(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int, 
                 decoder_config: Optional[DecoderConfig] = None,
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
        
        self.decoder_config = decoder_config
        self.encoder_only = decoder_config is None or num_decoder_layers == 0

        if not self.encoder_only:
            self.decoder = DecoderWrapper(hidden_dim, num_decoder_layers, self.token_emb, self.pos_emb, block_config, decoder_config)
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