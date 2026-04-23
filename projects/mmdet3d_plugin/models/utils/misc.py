import torch
import torch.nn as nn
import numpy as np
from mmdet.core import bbox_xyxy_to_cxcywh
from mmdet.models.utils.transformer import inverse_sigmoid
from peft import LoraConfig, get_peft_model

def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:]) 
    return memory * prev_exist
    
def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape
        
        view_shape = [1 for _ in range(len(feat_shape))] 
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)
        
        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def apply_ltrb(locations, pred_ltrb): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[..., 0] = (locations[..., 0] - pred_ltrb[..., 0])# x1
        pred_boxes[..., 1] = (locations[..., 1] - pred_ltrb[..., 1])# y1
        pred_boxes[..., 2] = (locations[..., 0] + pred_ltrb[..., 2])# x2
        pred_boxes[..., 3] = (locations[..., 1] + pred_ltrb[..., 3])# y2
        min_xy = pred_boxes[..., 0].new_tensor(0)
        max_xy = pred_boxes[..., 0].new_tensor(1)
        pred_boxes  = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
        pred_boxes  = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)
        pred_boxes = bbox_xyxy_to_cxcywh(pred_boxes)


        return pred_boxes    

def apply_center_offset(locations, center_offset): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        centers_2d = torch.zeros_like(center_offset)
        locations = inverse_sigmoid(locations)
        centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x1
        centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y1
        centers_2d = centers_2d.sigmoid()

        return centers_2d

@torch.no_grad()
def locations(features, stride, pad_h, pad_w):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """
        h, w = features.size()[-2:]
        device = features.device
        
        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        
        locations = locations.reshape(h, w, 2)
        
        return locations



def gaussian_2d(shape, sigma=1.0):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

class SELayer_Linear(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
        

class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, with_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.with_ln = with_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.with_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.with_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out


def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points

def transform_reference_points_lane(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1).unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points


def load_uni_model(base_model, use_lora, frozen, 
                   finetune_emb=False, new_token_ids=None,alpha=16,r=16, 
                   frozen_vit=False, 
                   add_new_modules=False, 
                   new_modules_list=None,
                   dtype=torch.float32, use_ckpt=True,
                   full_ft_attn=False,
                   full_ft=False, random_init=False,
                   random_init_list=None,
                   small_ffn=False,
                   only_e2e_head=False,
                   use_e2e_head=False,
                   use_det_head=False,
                   use_map_head=False,
                   same_self_attn=True,
                   e2e_with_self_attn=False,
                   rm_temp_in_casual=False,
                   use_layers=None,
                   ):

    assert (use_lora and full_ft) == False, "LoRA and full fine-tuning cannot be used simultaneously."
    
    if use_lora:
        if "Intern" in base_model:
            # raise NotImplementedError
            from ..modeling_vlm.uni_modeling_internvl_chat import UniInternVLChatModel     
            from transformers.utils import is_flash_attn_2_available

            model = UniInternVLChatModel.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                # attn_implementation="flash_attention_2",
                ignore_mismatched_sizes=True,
                # device_map='cuda', # cuda cause BUG
                trust_remote_code=True,
            )
            
        
        elif "VL" in base_model:
            from ..dense_heads.uni_qwen25vl_fa_v17 import LlavaQwen2VLForCausalLM as UniLlavaQwen25VLForCausalLM_FA
            from transformers.utils import is_flash_attn_2_available
            assert is_flash_attn_2_available()
            model = UniLlavaQwen25VLForCausalLM_FA.from_pretrained(base_model,
                            # torch_dtype=torch.float16,
                            torch_dtype=torch.bfloat16,
                            # torch_dtype=torch.float32,
                            attn_implementation="flash_attention_2",
                            ignore_mismatched_sizes=True,
                            # device_map='auto')
                            # device_map='cpu')
                            device_map='cuda')
    else:
        if "Intern" in base_model:
            raise NotImplementedError
        
        elif "VL" in base_model:
            from ..dense_heads.uni_qwen25vl_fa_v17 import LlavaQwen2VLForCausalLM as UniLlavaQwen25VLForCausalLM_FA
            from transformers.utils import is_flash_attn_2_available
            assert is_flash_attn_2_available()
            model = UniLlavaQwen25VLForCausalLM_FA.from_pretrained(base_model,
                            # torch_dtype=torch.float16, # 似乎只有float16能用
                            # torch_dtype=torch.float32,
                            torch_dtype=torch.bfloat16,
                            attn_implementation="flash_attention_2",
                            # device_map='auto')
                            # device_map='cpu')
                            device_map='cuda')
        else:
            raise NotImplementedError

    model.set_temp_casual(rm_temp_in_casual=rm_temp_in_casual)

    if dtype == torch.float32:
        model.replace_vision_encoder_to_flash_attention()
            
    if add_new_modules:
        try:
            model.create_new_modules(
                use_layers=use_layers,
                use_e2e_head=use_e2e_head,
                use_det_head=use_det_head,
                use_map_head=use_map_head,
                same_self_attn=same_self_attn,
                e2e_with_self_attn=e2e_with_self_attn,
            )
        except:
            import ipdb; ipdb.set_trace()
    
    if use_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    
    if use_lora:
        if small_ffn:
            assert random_init == True, "small_ffn must use random_init"
            model.resize_small_ffn()
        
        if random_init:
            model.init_llm_weights(random_init_list)
            target_modules = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
        else:
            target_modules = ('q_proj', 'k_proj', 'v_proj', 'o_proj')

        peft_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM")
        
        model = get_peft_model(model, peft_config)
        # 这是只有 LoRA 参数的 requires_grad=True 
        if random_init:# v6以及之后的版本不需要重新初始化 
            for n, p in model.named_parameters():
                if any([random_module in n and 'base_model.model.model.layers' in n for random_module in random_init_list]):
                    p.requires_grad = True

        if add_new_modules:
            for n, p in model.named_parameters():
                # qwen-vl
                if any([new_module in n and 'base_model.model.model.layers' in n for new_module in new_modules_list]):
                    p.requires_grad = True
            
                # internvl
                if any([new_module in n and 'base_model.model.language_model.model.layers' in n for new_module in new_modules_list]):
                    p.requires_grad = True
                    
        if full_ft_attn:
            import re
            for n, p in model.named_parameters():
                # qwen-vl
                m = re.search(r'base_model\.model\.model\.layers\.(\d+)\b', n)
                # if 'self_attn' in n and re.search(r'base_model\.model\.model\.layers\.[0-5]\b', n):
                if 'self_attn' in n and m and (int(m.group(1)) in use_layers):
                    p.requires_grad = False if 'lora' in n else True
                    
                # internvl
                m = re.search(r'base_model\.model\.language_model\.model\.layers\.(\d+)\b', n)
                if 'self_attn' in n and m and (int(m.group(1)) in use_layers):
                    p.requires_grad = False if 'lora' in n else True
        
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(dtype)
            
    elif full_ft:
        raise NotImplementedError
               
    return model

