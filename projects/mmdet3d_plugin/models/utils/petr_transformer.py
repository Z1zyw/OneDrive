# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
from mmdet.models.utils.builder import TRANSFORMER
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as cp
from .attention import FlashMHA
import warnings



from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2, repeat_kv
import math
class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = embed_dims
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_kv_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attention_dropout=dropout
        
    def forward(
        self,
        hidden_states: torch.Tensor, # query
        key,
        value,
        query_pos,
        key_pos,
        attn_mask=None,
        # w_rope=False
    ):
        bsz, q_len, _ = hidden_states.size()
        _, kv_len, _ = key.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(key)
        value_states = self.v_proj(value)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, -1, self.head_dim).transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if key_pos is not None:
            key_states = key_states + key_pos.view(bsz, kv_len, -1, self.head_dim).transpose(1, 2)
        if query_pos is not None:
            query_states = query_states + query_pos.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        
        return attn_output + hidden_states, attn_weights

from mmcv.runner import BaseModule
class MultiHeadAttentionwDropout(nn.Module):

    def __init__(self, embed_dims, num_heads, dropout, flash_attn):
        super().__init__()
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._dropout = dropout
        self.flash_attn = flash_attn
        if flash_attn:
            self.attn = FlashMHA(embed_dims, num_heads, dropout, dtype=torch.float16, device='cuda')
        else:
            self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.proj_drop = nn.Dropout(dropout)

        self._count = 0

    def forward(self, 
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_mask):
        """ Forward function for multi-head attention
        Args:
            query: shape [num_query, batch_size, embed_dims]
            key: shape [num_key, batch_size, embed_dims]
            value: shape [num_value, batch_size, embed_dims]
            query_pos: shape [num_query, batch_size, embed_dims]
            key_pos: shape [num_key, batch_size, embed_dims]
        """
        if query_pos is not None:
            query_w_pos = query + query_pos
        else:
            query_w_pos = query
        if key_pos is not None:
            key_w_pos = key + key_pos
        else:
            key_w_pos = key

        if self.flash_attn:
            out, attn = self.attn(query_w_pos, key_w_pos, value)
        else:
            out, attn = self.attn(query_w_pos, key_w_pos, value, attn_mask=attn_mask)
        out = self.proj_drop(out)

        return out + query, attn


# Feed-forward Network
class FFN(nn.Module):

    def __init__(self, embed_dims, feedforward_dims, dropout):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dims, embed_dims),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self._layers(x) + x

# We use the same decoder as StreamPETR
# which consists of:
# 1. Multi-head Self-attention 
# 2. LayerNorm
# 3. Multi-head Cross-attention
# 4. LayerNorm
# 5. Feed-forward Network
# 6. LayerNorm

class PETRTransformerDecoderLayer(nn.Module):

    def __init__(self, 
                 embed_dims, 
                 num_heads, 
                 feedforward_dims, 
                 dropout=0.1,
                 flash_attn=True,
                 mqa=False, # mqa_cross_attention
                 ):
        super().__init__()
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._feedforward_dims = feedforward_dims

        self.transformer_layers = nn.ModuleList()
        # 1. Multi-head Self-attention
        self.transformer_layers.append(
            MultiHeadAttentionwDropout(embed_dims, num_heads, dropout, False)
        )
        # 2. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )
        # 3. Multi-head Cross-attention
        self.transformer_layers.append(
            MultiHeadAttentionwDropout(embed_dims, num_heads, dropout, flash_attn) if not mqa else
            MultiQueryAttention(embed_dims, num_heads, num_kv_heads=2, dropout=dropout)
        )
        # 4. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )
        # 5. Feed-forward Network
        self.transformer_layers.append(
            FFN(embed_dims, feedforward_dims, dropout))
        # 6. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )
        

    def forward(self, query, key, query_pos, key_pos, attn_mask, temp_memory=None, temp_pos=None):
        """ Forward function for transformer decoder layer
        Args:
            query: shape [num_query, batch_size, embed_dims]
            key: shape [num_key, batch_size, embed_dims]
            value: shape [num_value, batch_size, embed_dims]
            query_pos: shape [num_query, batch_size, embed_dims]
            key_pos: shape [num_key, batch_size, embed_dims]
            attn_mask: shape [batch_size, num_query, num_key]
        """
        # TODO: maybe we shouldn't use hard-code layer here
        # TODO: add temporal query here
        # 1. Multi-head Self-attention (between queries)
        if temp_memory is not None:
            temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
            temp_pos = torch.cat([query_pos, temp_pos], dim=1)
        else:
            temp_key = temp_value = query
            temp_pos = query_pos
        # 通过 attn 进行时序融合：q: frame_t, k & v: [frame_t, memory]
        
        
        query, attn0 = self.transformer_layers[0](query, temp_key, temp_value, query_pos, temp_pos, attn_mask=attn_mask)
        # 2. LayerNorm
        query = self.transformer_layers[1](query)
        # 3. Multi-head Cross-attention (between queries and keys)
        query, attn1 = self.transformer_layers[2](query, key, key, query_pos, key_pos, attn_mask=None)
        # 4. LayerNorm
        query = self.transformer_layers[3](query)
        # 5. Feed-forward Network
        query = self.transformer_layers[4](query)
        # 6. LayerNorm
        query = self.transformer_layers[5](query)

        return query


class ConstructiveTransformerLayer(nn.Module):

    def __init__(self, 
                 embed_dims, 
                 num_heads, 
                 feedforward_dims, 
                 dropout=0.1,
                 flash_attn=True,
                 ):
        super().__init__()
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._feedforward_dims = feedforward_dims

        self.transformer_layers = nn.ModuleList()
        # 1. Multi-head Self-attention
        # self.transformer_layers.append(
        #     MultiHeadAttentionwDropout(embed_dims, num_heads, dropout, False)
        # )
        # # 2. LayerNorm
        # self.transformer_layers.append(
        #     nn.LayerNorm(embed_dims)
        # )
        # 3. Multi-head Cross-attention
        self.transformer_layers.append(
            MultiHeadAttentionwDropout(embed_dims, num_heads, dropout, flash_attn)
        )
        # 4. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )
        # 5. Feed-forward Network
        self.transformer_layers.append(
            FFN(embed_dims, feedforward_dims, dropout))
        # 6. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )
        

    def forward(self, query, key, query_pos, key_pos, attn_mask=None, temp_memory=None, temp_pos=None):
        """ Forward function for transformer decoder layer
        Args:
            query: shape [num_query, batch_size, embed_dims]
            key: shape [num_key, batch_size, embed_dims]
            value: shape [num_value, batch_size, embed_dims]
            query_pos: shape [num_query, batch_size, embed_dims]
            key_pos: shape [num_key, batch_size, embed_dims]
            attn_mask: shape [batch_size, num_query, num_key]
        """
        # TODO: maybe we shouldn't use hard-code layer here
        # TODO: add temporal query here
        # 1. Multi-head Self-attention (between queries)
        # if temp_memory is not None:
        #     temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
        #     temp_pos = torch.cat([query_pos, temp_pos], dim=1)
        # else:
        #     temp_key = temp_value = query
        #     temp_pos = query_pos

        # query, attn0 = self.transformer_layers[0](query, temp_key, temp_value, query_pos, temp_pos, attn_mask=attn_mask)
        # # 2. LayerNorm
        # query = self.transformer_layers[1](query)
        # 3. Multi-head Cross-attention (between queries and keys)
        query, attn1 = self.transformer_layers[0](query, key, key, query_pos, key_pos, attn_mask=None)
        # 4. LayerNorm
        query = self.transformer_layers[1](query)
        # 5. Feed-forward Network
        query = self.transformer_layers[2](query)
        # 6. LayerNorm
        query = self.transformer_layers[3](query)

        return query


@TRANSFORMER.register_module()
class PETRTransformerDecoder(nn.Module):
    def __init__(self, 
                 num_layers,
                 embed_dims,
                 num_heads,
                 feedforward_dims,
                 dropout,
                 with_cp=False,
                 flash_attn=True,
                 mqa=False, 
                 prenorm=False
                ):
        super().__init__()
        self._num_layers = num_layers
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._feedforward_dims = feedforward_dims
        self._dropout = dropout
        self._with_cp = with_cp
        self._layers = nn.ModuleList()
        for _ in range(num_layers):
            self._layers.append(
                PETRTransformerDecoderLayer(
                    embed_dims,
                    num_heads,
                    feedforward_dims,
                    dropout,
                    flash_attn=flash_attn,
                    mqa=mqa,
                ) if not prenorm else
                PETRTransformerDecoderLayerPreNorm(
                    embed_dims,
                    num_heads,
                    feedforward_dims,
                    dropout,
                    flash_attn=flash_attn,
                    mqa=mqa,
                )
            )

    def forward(self, query, key, query_pos=None, key_pos=None, attn_mask=None, temp_memory=None, temp_pos=None):
        """ Forward function for transformer decoder
        Args:
            query: shape [num_query, batch_size, embed_dims]
            key: shape [num_key, batch_size, embed_dims]
            value: shape [num_value, batch_size, embed_dims]
            query_pos: shape [num_query, batch_size, embed_dims]
            key_pos: shape [num_key, batch_size, embed_dims]
            attn_mask: shape [batch_size, num_query, num_key]
        """
        return_val = []
        for layer in self._layers:
            if self._with_cp and self.training:
                query = cp.checkpoint(layer, query, key, query_pos, key_pos, attn_mask, temp_memory, temp_pos)
            else:
                query = layer(query, key, query_pos, key_pos, attn_mask, temp_memory, temp_pos)
            return_val.append(query)
        return torch.stack(return_val, dim=0)

@TRANSFORMER.register_module()
class PETRTemporalTransformer(nn.Module):
    def __init__(self, 
                 input_dimension,
                 output_dimension,
                 query_number=32,
                 num_layers=6,
                 embed_dims=256,
                 num_heads=8,
                 feedforward_dims=2048,
                 dropout=0.0,
                 with_cp=False,
                 flash_attn=True,
                 mqa=False,
                 prenorm=False,
                 ):
        
        super().__init__()
        assert output_dimension % embed_dims == 0, "output dimension (language model) must be divisible by the embed dimension"

        # self.query_embedding = nn.Embedding(query_number, embed_dims) # learnable queries
        
        self.input_dimension = embed_dims
        self.output_dimension = output_dimension

        self.query_decoder = PETRTransformerDecoder(
                                            num_layers=num_layers,
                                            embed_dims=embed_dims,
                                            num_heads=num_heads,
                                            feedforward_dims=feedforward_dims,
                                            dropout=dropout,
                                            with_cp=with_cp,
                                            flash_attn=flash_attn,
                                            mqa=mqa,
                                            prenorm=prenorm
                                            )


        # convert with linear layer
        # self.output_projection = nn.Linear(embed_dims, output_dimension)
        # according to shihao, merge the multiple tokens into one to expand dimension

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1 and m.weight.requires_grad:
                nn.init.xavier_uniform_(m.weight)


    def forward(self, query, key, query_pos=None, key_pos=None, attn_mask=None, temp_memory=None, temp_pos=None):
        """ Forward function for transformer decoder
        Args:
            vision_tokens: shape [bs, sequence_length, embed_dims]
        Output:
            re-sampled token sequences: [bs, num_queries, embed_dims]
        """

        out = self.query_decoder(query, key, query_pos, key_pos, attn_mask, temp_memory, temp_pos) # feature from the last layer

        return out
    

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2Model, Qwen2MLP, Qwen2RMSNorm
class MultiQueryAttentionPreNorm(BaseModule):
    def __init__(self, embed_dims, num_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = embed_dims
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_kv_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attention_dropout=dropout
        
        self.norm = Qwen2RMSNorm(embed_dims, eps=1e-6)
    
    def forward(
        self,
        hidden_states: torch.Tensor, # query
        key,
        value,
        query_pos,
        key_pos,
        attn_mask=None,
        # w_rope=False
    ):
        bsz, q_len, _ = hidden_states.size()
        _, kv_len, _ = key.size()

        query_states = self.q_proj(self.norm(hidden_states))
        key_states = self.k_proj(key)
        value_states = self.v_proj(value)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, -1, self.head_dim).transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if key_pos is not None:
            key_states = key_states + key_pos.view(bsz, kv_len, -1, self.head_dim).transpose(1, 2)
        if query_pos is not None:
            query_states = query_states + query_pos.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        
        return attn_output + hidden_states, attn_weights
        
class FFNPreNorm(nn.Module):

    def __init__(self, embed_dims, feedforward_dims, dropout):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dims, embed_dims),
            nn.Dropout(dropout)
        )
        self.norm = Qwen2RMSNorm(embed_dims, eps=1e-6)

    def forward(self, x):
        return self._layers(self.norm(x)) + x
    
class QwenFFNPreNorm(BaseModule):
    def __init__(self, embed_dims, feedforward_dims, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dims, feedforward_dims, bias=False)
        self.up_proj = nn.Linear(embed_dims, feedforward_dims, bias=False)
        self.down_proj = nn.Linear(feedforward_dims, embed_dims, bias=False)
        # self.act_fn = ACT2FN[config.hidden_act]
        self.act_fn = nn.SiLU()
        
        self.norm = Qwen2RMSNorm(embed_dims, eps=1e-6)
    
    def init_weights(self):
        pass

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(self.norm(x))) * self.up_proj(self.norm(x)))
        return down_proj + x



class PETRTransformerDecoderLayerPreNorm(nn.Module):

    def __init__(self, 
                 embed_dims, 
                 num_heads, 
                 feedforward_dims, 
                 dropout=0.1,
                 flash_attn=True,
                 mqa=False, # mqa_cross_attention
                 ):
        super().__init__()
        assert mqa
        self._embed_dims = embed_dims
        if embed_dims == 896:
            num_heads = 14
            self._num_heads = 14
        else:
            raise NotImplementedError
        # self._num_heads = num_heads
        self._feedforward_dims = feedforward_dims
        
        
        self.transformer_layers = nn.ModuleList()
        # 1. Multi-head Self-attention
        self.transformer_layers.append(
            MultiHeadAttentionwDropout(embed_dims, num_heads, dropout, False)
        )
        # 2. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims) # post norm for self-attention
        )
        # 3. Multi-head Cross-attention
        self.transformer_layers.append(
            MultiHeadAttentionwDropout(embed_dims, num_heads, dropout, flash_attn) if not mqa else
            MultiQueryAttentionPreNorm(embed_dims, num_heads, num_kv_heads=2, dropout=dropout)
        )
        # 5. Feed-forward Network
        self.transformer_layers.append(
            # FFNPreNorm(embed_dims, feedforward_dims, dropout))
            QwenFFNPreNorm(embed_dims, feedforward_dims, dropout))
        

    def forward(self, query, key, query_pos, key_pos, attn_mask, temp_memory=None, temp_pos=None):
        """ Forward function for transformer decoder layer
        Args:
            query: shape [num_query, batch_size, embed_dims]
            key: shape [num_key, batch_size, embed_dims]
            value: shape [num_value, batch_size, embed_dims]
            query_pos: shape [num_query, batch_size, embed_dims]
            key_pos: shape [num_key, batch_size, embed_dims]
            attn_mask: shape [batch_size, num_query, num_key]
        """
        # TODO: maybe we shouldn't use hard-code layer here
        # TODO: add temporal query here
        # 1. Multi-head Self-attention (between queries)
        if temp_memory is not None:
            temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
            temp_pos = torch.cat([query_pos, temp_pos], dim=1)
        else:
            temp_key = temp_value = query
            temp_pos = query_pos
        # 通过 attn 进行时序融合：q: frame_t, k & v: [frame_t, memory]
        query, attn0 = self.transformer_layers[0](query, temp_key, temp_value, query_pos, temp_pos, attn_mask=attn_mask)
        
        # 2. LayerNorm
        query = self.transformer_layers[1](query)
        # 3. Multi-head Cross-attention (between queries and keys)
        query, attn1 = self.transformer_layers[2](query, key, key, query_pos, key_pos, attn_mask=None)
        # # 4. LayerNorm
        # query = self.transformer_layers[3](query)
        # 5. Feed-forward Network
        query = self.transformer_layers[3](query)
        return query