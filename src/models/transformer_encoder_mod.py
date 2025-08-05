import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class ModificationsConfig():
    condition_every: bool = False
    replace_cross_attn: bool = False
    remove_self_attn: bool = False
    replace_self_attn: bool = False 



class TransformerEncoderLayerModified(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0.1, activation = F.relu, layer_norm_eps = 0.00001, batch_first = False, norm_first = False, bias = True, 
                 modifications_config: ModificationsConfig = ModificationsConfig(), device=None, dtype=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype)

        self.modifications_config = modifications_config

        # Condition block replaces cross-attention
        if modifications_config.condition_every or modifications_config.replace_cross_attn:
            self.norm_cond = nn.LayerNorm(d_model)

        # Replacing cross-attention with the same parameter budget (only W_o and W_v, since W_q and W_k are not being used)
        if modifications_config.replace_cross_attn:
            self.dropout1_cond = nn.Dropout(dropout)
            self.dropout2_cond = nn.Dropout(dropout)
            self.lin1_cond = nn.Linear(d_model, d_model)
            self.lin2_cond = nn.Linear(d_model, d_model)

        if modifications_config.remove_self_attn:
            del self.norm1
            del self.self_attn
        
        # Replacing self-attention by an ffn block with the same parameter budget
        if modifications_config.replace_self_attn:
            self.norm_extra = nn.LayerNorm(d_model)
            self.dropout1_extra = nn.Dropout(dropout)
            self.dropout2_extra = nn.Dropout(dropout)
            self.lin1_extra = nn.Linear(d_model, 2*d_model)
            self.lin2_extra = nn.Linear(2*d_model, d_model)


    def _cond_block(self, x, x_condition):
        if self.modifications_config.condition_every:
            x = x + x_condition

        if self.modifications_config.replace_cross_attn:
            x = self.lin2_cond(self.dropout1_cond(self.activation(self.lin1_cond(x))))
        return x
    
    def _extra_ffn_block(self, x):
        x = self.lin2_extra(self.dropout1_extra(self.activation(self.lin1_extra(x))))
        return self.dropout2_extra(x)

        

    def forward(self, src, src_condition=None, mask=None, src_key_padding_mask=None, is_causal=None):
        x = src

        if self.modifications_config.condition_every or self.modifications_config.replace_cross_attn:
            x = x + self._cond_block(self.norm_cond(x), src_condition)

        if not self.modifications_config.remove_self_attn:
            x = x + self._sa_block(self.norm1(x), mask, src_key_padding_mask, is_causal)    
        elif self.modifications_config.replace_self_attn:
            x = x + self._extra_ffn_block(self.norm_extra(x)) 

        x = x + self._ff_block(self.norm2(x))
        return x
    
class TransformerEncoderModified(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm = None, enable_nested_tensor = True, mask_check = True):
        super().__init__(encoder_layer, num_layers, norm, enable_nested_tensor, mask_check)

    def forward(self, src, src_condition, mask=None, src_key_padding_mask=None, is_causal=None):
        x = src
        for mod in self.layers:
            x = mod(x, src_condition, mask, src_key_padding_mask, is_causal)

        return x
        
