from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, repeat_kv, apply_rotary_pos_emb, Qwen2RMSNorm
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import torch
import torch.nn as nn
from typing import Optional
from ..utils.token_spec import TokenSpec
from typing import Tuple
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from projects.mmdet3d_plugin.models.utils.petr_transformer import MultiHeadAttentionwDropout, FlashMHA

class SafeMultiHeadAttentionwDropout(nn.Module):

    def __init__(self, embed_dims, num_heads, dropout, flash_attn):
        super().__init__()
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._dropout = dropout
        self.flash_attn = flash_attn
        if flash_attn:
            self.attn = FlashMHA(embed_dims, num_heads, dtype=torch.float16, device='cuda')
        else:
            self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.proj_drop = nn.Dropout(dropout)
        
        self._count = 0
    
    # forward in float32 to avoid nan
    # @ force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'key_pos', 'attn_mask'))
    def forward(self,
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_mask):
        """ Forward function for multi-head attention
        batch_first: True
        # Args:
        #     query: shape [num_query, batch_size, embed_dims]
        #     key: shape [num_key, batch_size, embed_dims]
        #     value: shape [num_value, batch_size, embed_dims]
        #     query_pos: shape [num_query, batch_size, embed_dims]
        #     key_pos: shape [num_key, batch_size, embed_dims]
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

        return out, attn


class QueryLayerWiseSelfFlashAttnQwenWithResidualNorm(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.task_self_attn = SafeMultiHeadAttentionwDropout(
            config.hidden_size,
            config.num_attention_heads,
            dropout=0.0,
            flash_attn=True,
        )
        self.task_norm = Qwen2RMSNorm(config.hidden_size, 1e-6)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        images_embeds: Optional[torch.FloatTensor] = None,
        images_embeds_pos_3d: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # flag for valid token
        token_spec: Optional[TokenSpec] = None, # flag for bidirectional token
        position_embeds_3d: Optional[torch.FloatTensor] = None,
        plugin_position_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None, # Optional[Cache]
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ):
        output = hidden_states.clone()
        bsz, q_len, dim = hidden_states.size()
        assert token_spec is not None, "token_spec is required for task_self_attn"
        
        query_slice = token_spec.is_det_query | token_spec.is_map_query
            
        kv_slice = token_spec.is_det_token | token_spec.is_map_token
        
        dtype = self.task_self_attn.attn.out_proj.weight.dtype
        task_query = hidden_states[query_slice, :].contiguous().to(dtype).reshape(bsz, -1, dim)
        
        residual = task_query
        task_query = self.task_norm(task_query)

        task_query_pos = plugin_position_embeds[query_slice, :].contiguous().to(dtype).reshape(bsz, -1, dim) if plugin_position_embeds is not None else None
        task_kv = hidden_states[kv_slice, :].contiguous().to(dtype).reshape(bsz, -1, dim)
        task_kv_pos = plugin_position_embeds[kv_slice, :].contiguous().to(dtype).reshape(bsz, -1, dim) if plugin_position_embeds is not None else None

        task_query, task_attn_mask = self.task_self_attn(
            query=task_query,
            key=task_kv,
            value=task_kv,
            query_pos=task_query_pos,
            key_pos=task_kv_pos,
            attn_mask=None
        )
        task_query = task_query + residual 

        output_new = output.clone()
        output_new[query_slice] = task_query.to(hidden_states.dtype).reshape(-1, dim)

        return output_new, task_attn_mask

class MixedAttention(Qwen2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        images_embeds: Optional[torch.FloatTensor] = None,
        images_embeds_pos_3d: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # flag for valid token
        token_spec: Optional[TokenSpec] = None, # flag for bidirectional token
        position_embeds_3d: Optional[torch.FloatTensor] = None,
        plugin_position_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None, # Optional[Cache]
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        rm_temp_in_casual: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        output = hidden_states.clone()
        
        # if token_spec is not None and self.layer_idx < MAX_LAYERS_NUM:
        #     output_task = output.clone()
        # else:
        #     output_task = None
            
        bsz, q_len, dim = hidden_states.size()

        if rm_temp_in_casual and token_spec is not None and q_len > 1: 
            attn_slice = ~token_spec.is_temporal_token
            hidden_states = hidden_states[attn_slice].contiguous().reshape(bsz, -1, dim)
            bsz, q_len, dim = hidden_states.size()
            # attn_slice_repeated = attn_slice[None].repeat(3, 1, 1)
            new_position_embeds_cos = position_embeddings[0][attn_slice].contiguous().reshape(bsz, q_len, -1)
            new_position_embeds_sin = position_embeddings[1][attn_slice].contiguous().reshape(bsz, q_len, -1)
            position_embeddings = (new_position_embeds_cos, new_position_embeds_sin)
            
            plugin_position_embeds = plugin_position_embeds[attn_slice].contiguous().reshape(bsz, -1, dim) if plugin_position_embeds is not None else None
            
            # if bsz > 1:
            attention_mask = attention_mask[attn_slice].contiguous().reshape(bsz, -1) if attention_mask is not None else None
            
        
        _input_dtype = hidden_states.dtype
        _dtype = self.q_proj.weight.dtype 
        hidden_states = hidden_states.to(_dtype)
        query_states = self.q_proj(hidden_states).to(_input_dtype)
        key_states = self.k_proj(hidden_states).to(_input_dtype)
        value_states = self.v_proj(hidden_states).to(_input_dtype)
        hidden_states = hidden_states.to(_input_dtype)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) 

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
            
        if query_states.shape[2] > 1: 
            # > 1 means firstly generation or training
            assert plugin_position_embeds is not None
            pos_3d = plugin_position_embeds.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            query_states = query_states + pos_3d
            key_states = key_states + pos_3d
        else:
            # == 1 means generation with kv_cache, and no 3d position embedding added
            pass
            
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        output_ = attention_interface(
            self,
            query_states.to(torch.bfloat16),
            key_states.to(torch.bfloat16),
            value_states.to(torch.bfloat16),
            attention_mask,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )[0]

        output_ = output_.reshape(bsz, q_len, dim).contiguous()
        _dtype = self.o_proj.weight.dtype
        _input_dtype = output_.dtype
        output_ = self.o_proj(output_.to(_dtype)).to(_input_dtype)
        
        if rm_temp_in_casual and token_spec is not None and q_len > 1:
            output[attn_slice] = output_.flatten(0, 1).to(hidden_states.dtype)
        else:
            output = output_.to(hidden_states.dtype)

        
        if not output_attentions:
            attn_weights = None
        
        return output, attn_weights, past_key_value, output