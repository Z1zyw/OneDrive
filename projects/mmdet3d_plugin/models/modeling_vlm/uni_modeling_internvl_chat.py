# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from functools import partial
from typing import List, Optional, Tuple, Union, Dict

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn

from .modeling_internvl_chat import InternVLChatModel

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache, Cache

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2Model, Qwen2MLP

from .mixed_attn import MixedAttention, QueryLayerWiseSelfFlashAttnQwenWithResidualNorm
from ..utils.token_spec import TokenSpec, init_from_metas as token_spec_init_from_metas

from dataclasses import dataclass

logger = logging.get_logger(__name__)


# Hack
CONTEXT_ID = 151655

# 2026.2.26
HACK_RES = False
HACK_MV_SELF_ATTN = False



def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


@dataclass
class MixedLMOutputWithPast(BaseModelOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    task_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    token_spec: Optional[TokenSpec] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    # rope_deltas: Optional[torch.FloatTensor] = None


class MixAttnDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.self_attn = MixedAttention(config, layer_idx)
    
    def forward(
        self, 
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.FloatTensor] = None,
        plugin_position_embeds: Optional[torch.FloatTensor] = None,
        token_spec: Optional[TokenSpec] = None,
        rm_temp_in_casual: bool = False,
        **kwargs,
    ):
        bsz, q_len, dim = hidden_states.size()
        
        if HACK_MV_SELF_ATTN:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        
        if hasattr(self, 'query_self_attn') and hidden_states.size(1) > 1:
            hidden_states, _ = self.query_self_attn(
                hidden_states=hidden_states,
                token_spec=token_spec,
                plugin_position_embeds=plugin_position_embeds,
            )
        
        if not HACK_MV_SELF_ATTN:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        
        # Mixed Attention
        hidden_states, self_attn_weights, present_key_value, task_hidden_states = self.self_attn(
            hidden_states=hidden_states,
            token_spec=token_spec,
            plugin_position_embeds=plugin_position_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            rm_temp_in_casual=rm_temp_in_casual,
            **kwargs
        )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # sub forward mlp for task queries
        if (hasattr(self, 'query_mlp_det') or hasattr(self, 'query_mlp_map') or hasattr(self, 'query_mlp_e2e')) \
                and hidden_states.shape[1] > 1:

            query_slice = token_spec.is_det_token | token_spec.is_map_token | token_spec.is_e2e_token
            if HACK_RES:
                hidden_states_dummy = hidden_states.clone()
                task_residual = residual
                task_hidden_states = hidden_states
            else:
                hidden_states_dummy = task_hidden_states.clone()
                task_residual = task_hidden_states
                task_hidden_states = self.input_layernorm(task_hidden_states)
            bzs, _, hdim = hidden_states.shape

            if hasattr(self, 'query_mlp_det'):
                det_slice = token_spec.is_det_token
                mlp_dtype = self.query_mlp_det.gate_proj.weight.dtype
                hidden_states_det = self.query_mlp_det(task_hidden_states[det_slice].contiguous().to(mlp_dtype)).to(task_hidden_states.dtype)
                hidden_states_det = task_residual[det_slice].contiguous() + hidden_states_det
                hidden_states_dummy[det_slice] = hidden_states_det
                
            if hasattr(self, 'query_mlp_map'):
                map_slice = token_spec.is_map_token
                mlp_dtype = self.query_mlp_map.gate_proj.weight.dtype
                hidden_states_map = self.query_mlp_map(task_hidden_states[map_slice].contiguous().to(mlp_dtype)).to(task_hidden_states.dtype)
                hidden_states_map = task_residual[map_slice].contiguous() + hidden_states_map
                hidden_states_dummy[map_slice] = hidden_states_map
                
            if hasattr(self, 'query_mlp_e2e'):
                e2e_slice = token_spec.is_e2e_token
                mlp_dtype = self.query_mlp_e2e.gate_proj.weight.dtype
                hidden_states_e2e = self.query_mlp_e2e(task_hidden_states[e2e_slice].contiguous().to(mlp_dtype)).to(task_hidden_states.dtype)
                hidden_states_e2e = task_residual[e2e_slice].contiguous() + hidden_states_e2e
                hidden_states_dummy[e2e_slice] = hidden_states_e2e
                
            hidden_states_ = hidden_states_dummy[query_slice]
            hidden_states_ = hidden_states_.reshape(bzs, -1, hdim)
            
        else:
            hidden_states_ = None
            
        # hack fix bug: different dtype
        mlp_dtype = self.mlp.gate_proj.weight.dtype
        hidden_dtype = hidden_states.dtype
        
        hidden_states = self.mlp(hidden_states.to(mlp_dtype)).to(hidden_dtype)
        hidden_states = residual + hidden_states
        
        if (hasattr(self, 'query_mlp_det') or hasattr(self, 'query_mlp_map') or hasattr(self, 'query_mlp_e2e')) \
            and hidden_states.shape[1] > 1:
            hidden_states = hidden_states.clone()
            hidden_states[query_slice] = hidden_states_.flatten(0,1)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, hidden_states_
            

class UniInternLLM(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([MixAttnDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
    
    def set_temp_casual(self, rm_temp_in_casual=False):
        self.rm_temp_in_casual = rm_temp_in_casual
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds:  Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        plugin_position_embeds: Optional[torch.FloatTensor] = None,
        token_spec: Optional[TokenSpec] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        next_decoder_cache = None
        
        task_hidden_states = []

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs, task_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    plugin_position_embeds,
                    token_spec,
                    self.rm_temp_in_casual,
                )
            else:
                layer_outputs, task_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    plugin_position_embeds=plugin_position_embeds,
                    token_spec=token_spec,
                    rm_temp_in_casual=self.rm_temp_in_casual,
                    **flash_attn_kwargs,
                )
            if task_outputs is not None:
                task_hidden_states.append(task_outputs)
            
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        
        if len(self.layers) == 6:
            # HACK imple for test
            pass 
        else:
            hidden_states = self.norm(hidden_states)
            # pass # HACK for test

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return MixedLMOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            token_spec=token_spec,
            task_hidden_states=task_hidden_states
        )
        
        
class UniInternVLForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = UniInternLLM(config)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds:  Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        plugin_position_embeds: Optional[torch.FloatTensor] = None,
        token_spec: Optional[TokenSpec] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            plugin_position_embeds=plugin_position_embeds,
            token_spec=token_spec,  
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        if self.lm_head is not None:
            logits = self.lm_head(outputs.last_hidden_state.to(self.lm_head.weight.dtype))
        else:
            logits=None
        
        return MixedLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            
            token_spec=outputs.token_spec,
            task_hidden_states=outputs.task_hidden_states
        )

# 这个是最终模型
class UniInternVLChatModel(InternVLChatModel):
    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config, vision_model, language_model, use_flash_attn)
        self.language_model = UniInternVLForCausalLM(config.llm_config)
    
    def set_temp_casual(self, rm_temp_in_casual=False):
        self.language_model.model.set_temp_casual(rm_temp_in_casual)
    
    def init_weights(self):
        pass
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def replace_vision_encoder_to_flash_attention(self):
        pass 
    
    def create_new_modules(
        self, 
        use_layers=None,
        use_e2e_head=False,
        use_det_head=True, 
        use_map_head=False,
        same_self_attn=True,
        e2e_with_self_attn=False,
        only_e2e_head=False):
        assert use_layers is not None
        for i, layer in enumerate(self.language_model.model.layers):
            if i not in use_layers:
                continue
            if use_det_head or use_map_head:
                if same_self_attn:
                    layer.query_self_attn = QueryLayerWiseSelfFlashAttnQwenWithResidualNorm(self.language_model.model.config, i)
                else:
                    import ipdb; ipdb.set_trace()
                    # layer.query_self_attn = TaskDecoupledQueryLayerWiseSelfFlashAttnQwenWithResidualNorm(
                    #     self.model.config, i,
                    #     use_det=use_det_head,
                    #     use_map=use_map_head,
                    #     use_e2e=e2e_with_self_attn and use_e2e_head
                    # )
            if use_det_head:
                layer.query_mlp_det = Qwen2MLP(self.language_model.model.config)
                layer.query_mlp_det.apply(self._init_weights)
            if use_map_head:
                layer.query_mlp_map = Qwen2MLP(self.language_model.model.config)
                layer.query_mlp_map.apply(self._init_weights)
            if use_e2e_head and not only_e2e_head:
                layer.query_mlp_e2e = Qwen2MLP(self.language_model.model.config)
                layer.query_mlp_e2e.apply(self._init_weights)

        
    def get_rope_ids(self, input_ids, attention_mask, tasks_spec=None):
        if attention_mask is None:
            attention_mask_ = torch.ones_like(input_ids)
        else:
            attention_mask_ = attention_mask

        # if tasks_spec is not None:
        #     query_mask = tasks_spec.is_e2e_token | tasks_spec.is_det_token | tasks_spec.is_map_token
        #     attention_mask_[query_mask] = False
        #     query_mask = tasks_spec.is_begin_token
        #     attention_mask_[query_mask] = True
            
        position_ids = attention_mask_.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask_ == 0, 1)
        return position_ids
        
    
    def forward(
        self,
        # pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # dummy input to align PEFT
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images_embeds: Optional[torch.FloatTensor] = None,
        images_pos_embeds_3d: Optional[torch.FloatTensor] = None,
        tasks_metas: Optional[Dict] = None,
        tasks_embeds: Optional[torch.LongTensor] = None,
        tasks_pos_embeds: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        if tasks_embeds is not None:
            input_embeds = input_embeds.to(tasks_embeds.dtype)
        else:
            input_embeds = input_embeds.to(torch.float32)

        # vit_embeds = self.extract_feature(pixel_values)
        # vit_embeds = vit_embeds[image_flags == 1]
        # vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        # input_embeds = input_embeds.reshape(B * N, C)
        plugin_position_embeds = torch.zeros_like(input_embeds)
        token_spec = None

        # hack imple
        selected = (input_ids == CONTEXT_ID)
        
        if selected.sum() > 0: # 
            assert images_embeds is not None
            if tasks_embeds is not None:
                new_images_embeds = torch.cat([images_embeds, tasks_embeds], dim=1)
                images_embeds = new_images_embeds
            images_embeds = images_embeds.reshape(-1, C)
            assert selected.sum() == images_embeds.shape[0], f'selected sum: {selected.sum()}, images_embeds shape: {images_embeds.shape}'
            input_embeds[selected] = input_embeds[selected] * 0.0 + images_embeds
            
            # prepare token spec 
            token_spec = token_spec_init_from_metas(tasks_metas, input_ids, selected)
            
            # prepare RoPE
            if position_ids is None:
                # TODO: task embeddings use same ids
                position_ids = self.get_rope_ids(input_ids, attention_mask, token_spec)
            
            # prepare 3D pos embed and query pos
            if images_pos_embeds_3d is not None:
                _plugin_pos_embeds = images_pos_embeds_3d
                if tasks_pos_embeds is not None:
                    _plugin_pos_embeds = torch.cat([_plugin_pos_embeds, tasks_pos_embeds], dim=1)
                plugin_position_embeds[selected] = plugin_position_embeds[selected] * 0.0 + _plugin_pos_embeds.reshape(-1, C) 
                
                
        # input_embeds = input_embeds.reshape(B, N, C)
        # plugin_position_embeds = plugin_position_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            plugin_position_embeds=plugin_position_embeds,
            token_spec=token_spec,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and logits is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MixedLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            
            token_spec=outputs.token_spec,
            task_hidden_states=outputs.task_hidden_states
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            # pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            tasks_embeds: Optional[torch.FloatTensor] = None,
            tasks_pos_embeds: Optional[torch.FloatTensor] = None,
            tasks_metas: Optional[TokenSpec] = None,
            images_embeds: Optional[torch.FloatTensor] = None,
            images_pos_embeds_3d: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            # visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        if tasks_embeds is not None:
            input_embeds = input_embeds.to(tasks_embeds.dtype)
        else:
            input_embeds = input_embeds.to(torch.float32) # hack imple 
        B, N, C = input_embeds.shape
        # input_embeds = input_embeds.reshape(B * N, C)
        plugin_position_embeds = torch.zeros_like(input_embeds)
        token_spec = None

        # hack imple
        selected = (input_ids == CONTEXT_ID)
        
        if selected.sum() > 0: # 
            assert images_embeds is not None
            if tasks_embeds is not None:
                new_images_embeds = torch.cat([images_embeds, tasks_embeds], dim=1)
                images_embeds = new_images_embeds
            images_embeds = images_embeds.reshape(-1, C)
            assert selected.sum() == images_embeds.shape[0], f'selected sum: {selected.sum()}, images_embeds shape: {images_embeds.shape}'
            input_embeds[selected] = input_embeds[selected] * 0.0 + images_embeds
            
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            
            # prepare RoPE
            if position_ids is None:
                # TODO: task embeddings use same ids
                # TODO: for generate, input_ids delta
                position_ids = self.get_rope_ids(input_ids, attention_mask)
            
            # prepare 3D pos embed and query pos
            if images_pos_embeds_3d is not None:
                _plugin_pos_embeds = images_pos_embeds_3d
                if tasks_pos_embeds is not None:
                    _plugin_pos_embeds = torch.cat([_plugin_pos_embeds, tasks_pos_embeds], dim=1)
                plugin_position_embeds[selected] = plugin_position_embeds[selected] * 0.0 + _plugin_pos_embeds.reshape(-1, C) 
                
            # prepare token spec 
            token_spec = token_spec_init_from_metas(tasks_metas, input_ids, selected)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        
        outputs = self.language_model.generate(
            # input_ids=input_ids,

            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            plugin_position_embeds=plugin_position_embeds,
            token_spec=token_spec,
            output_hidden_states=output_hidden_states,
            
            use_cache=True,
            # generation_config=generation_config,
            
            # Hack, Maybe BUG
            # pad_token_id=151643,
            eos_token_id=151645,
            pad_token_id=151645,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
