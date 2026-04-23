# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear

import numpy as np
import os

@HEADS.register_module()
class UniEgoStatusHead(nn.Module):
    def __init__(
        self, 
        embed_dims=4096,
        in_channels=256,
        num_classes=1,
        input_len=74,
    ):
        super(UniEgoStatusHead, self).__init__()
        self.embed_dims = embed_dims

        embed_dims_2 = 256
        self.can_bus_embed = nn.Sequential(
            nn.Linear(input_len, embed_dims_2), # canbus + command + egopose
            nn.ReLU(),
            nn.Linear(embed_dims_2, embed_dims),
        )
        
    def get_query_embedding(self, batch_size, data):
        #TODO: temporal
        can_bus = data['can_bus'] # b, 13
        command = data['command'].unsqueeze(-1) # b, 1 
        
        tmp = torch.cat([command, can_bus], dim=-1) # b, 14
        memory_keep_len = 2
        memory_canbus = tmp.unsqueeze(1).repeat(1, memory_keep_len, 1).flatten(1) # b, 28
        
        ego_pose = data['ego_pose'].flatten(1) # b, 16
        memory_ego_pose = ego_pose.unsqueeze(1).repeat(1, memory_keep_len, 1).flatten(1) # b, 32
        
        can_bus_input = torch.cat([command, can_bus, memory_canbus, memory_ego_pose], dim=1)
        
        can_bus_embed = self.can_bus_embed(can_bus_input).unsqueeze(1) # b, 1, c
        
        return can_bus_input, can_bus_embed
        
    def get_targets(self):
        pass 
    
    def loss(self):
        pass 
        