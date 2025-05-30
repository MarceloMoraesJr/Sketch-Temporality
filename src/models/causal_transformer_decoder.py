#ADAPTED CODE FROM: https://github.com/alex-matton/causal-transformer-decoder/blob/master/causal_transformer_decoder/model.py
#Adaptations:
# - BATCH FIRST
# - PRE-NORMALIZATION

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable
from torch import Tensor


class CausalTransformerDecoder(nn.TransformerDecoder):
    """Implementation of a transformer decoder based on torch implementation but
    more efficient. The difference is that it doesn't need to recompute the
    embeddings of all the past decoded tokens but instead uses a cache to
    store them. This makes use of the fact that the attention of a decoder is
    causal, so new predicted tokens don't affect the old tokens' embedding bc
    the corresponding attention cells are masked.
    The complexity goes from seq_len^3 to seq_len^2.

    This only happens in eval mode.
    In training mode, teacher forcing makes these optimizations unnecessary. Hence the
    Decoder acts like a regular nn.TransformerDecoder (except that the attention tgt
    masks are handled for you).
    """
    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        output = tgt

        if self.training:
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

            return output, None
        
        batch_first = self.layers[0].self_attn.batch_first

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                if batch_first:
                    output = torch.cat([cache[i], output], dim=1)
                else:
                    output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            if batch_first:
                new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=2)
            else:
                new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
                new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        cross_attention: bool = True,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None):

        super().__init__(
            d_model, 
            nhead, 
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype)
        
        self.cross_attention = cross_attention

        # Delete cross attention attributes
        if not cross_attention:
            del self.multihead_attn
            del self.norm2
            del self.dropout2
    
    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            see CausalTransformerDecoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim or bsz x seq_len x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim or bsz x 1 x hidden_dim
        """

        if not self.cross_attention and \
        (memory is not None or memory_mask is not None or memory_key_padding_mask is not None):
            raise ValueError("Non cross-attention model: memory related arguments should be None")

        batch_first = self.self_attn.batch_first
        x = tgt

        if self.training:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                x.size(1) if batch_first else x.size(0), device=x.device)
        else:
            attn_mask = None

            
        if self.norm_first:
            query = self.norm1(x)
            key_value = query

            if not self.training:
                if batch_first:
                    query = query[:, -1:, :]
                    x = x[:, -1:, :]
                else:
                    query = query[-1:, :, :]
                    x = x[-1:, :, :]

            x = x + self._sa_block(
                query, key_value, attn_mask, tgt_key_padding_mask
            )
            if memory is not None:
                x = x + self._mha_block(
                    self.norm2(x),
                    memory,
                    memory_mask,
                    memory_key_padding_mask
                )
            x = x + self._ff_block(self.norm3(x))
        else:
            query = key_value = x

            if not self.training:
                if batch_first:
                    query = query[:, -1:, :]
                    x = x[:, -1:, :]
                else:
                    query = query[-1:, :, :]
                    x = x[-1:, :, :]

            x = self.norm1(
                x + self._sa_block(query, key_value, attn_mask, tgt_key_padding_mask)
            )
            if memory is not None:
                x = self.norm2(
                    x
                    + self._mha_block(
                        x, memory, memory_mask, memory_key_padding_mask
                    )
                )
            x = self.norm3(x + self._ff_block(x))

        return x
    
    def _sa_block(
        self,
        query: Tensor,
        key_value: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:

        key_value = self.self_attn(
            query,
            key_value,
            key_value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(key_value)