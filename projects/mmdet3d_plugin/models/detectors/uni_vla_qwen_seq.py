# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# Modified by Shihao Wang
# ------------------------------------------------------------------------
# Modified by Yiwei Zhang
# ------------------------------------------------------------------------
# 12 -> 13: add TokenSpec for code refactoring
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from ...datasets.utils.constants import IGNORE_INDEX
from mmdet3d.models import builder
from transformers import AutoTokenizer, GenerationConfig, AutoProcessor
from torch import autocast

from ..utils.misc import load_uni_model as load_uni_model
from ..utils.positional_encoding import pos2posemb2d
import torch.nn as nn
import os
import json
import mmcv
import numpy as np
from projects.mmdet3d_plugin.models.utils.misc import MLN
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.datasets.utils import conversation as conversation_lib
import time
from projects.mmdet3d_plugin.datasets.utils.data_utils import tokenizer_image_traj_token
import pickle

from projects.mmdet3d_plugin.models.utils.token_spec import Modality, Role, Task, TokenSpec

import time 

# from qwen_vl_utils import process_vision_info

# def build_head(cfg):
#     """Build head."""
#     return MODELS.build(cfg)


def bbox3d2result(bboxes, scores, labels):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape (N, 7).
        scores (torch.Tensor): Scores with shape (N).
        labels (torch.Tensor): Labels with shape (N).

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
    """
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())

    return result_dict



@DETECTORS.register_module()
class UniVLAQwenVLSeq(MVXTwoStageDetector):
    """Petr3D."""
    def __init__(self,
                 save_path='./results_vlm/',
                 use_grid_mask=False,
                 embed_dims=256,
                 LID=True,
                 lora_rank=16,
                 use_qwen_visual=False,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 depth_num=64,
                 depth_start = 1,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_head=None,
                 map_head=None,
                 ego_status_embed=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 lm_head=None,
                 finetune_emb=False,
                 tokenizer=None,
                 tokenizer_max_length=2048,
                 train_cfg=None,
                 test_cfg=None,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 frozen=True,
                 only_train_e2e_head=False,
                 only_e2e_head=False,
                 e2e_with_self_attn=False,
                 same_self_attn=True,
                 rm_temp_in_casual=False,
                 
                 use_layers=[0,1,2,3,4,5],
                 
                 img_num=6,
                 llm_bf16=False,
                 
                 use_lora=False,
                 full_ft_attn=False,
                 full_ft=False,
                 random_init=False, # for full ft
                 random_init_list=['mlp'],
                 
                 small_ffn=False,
                 
                 frozen_vit=True,
                 pretrain_dist_token_path=None,
                 use_mapemb=True,
                 e2e_head=None,
                 
                 pretrained=None,
                 text_finetune=True,
                 text_loss_lambda=1.0,
                 use_same_query=True):
        super(UniVLAQwenVLSeq, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.use_qwen_visual = use_qwen_visual 
        self.save_path = save_path
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.only_train_e2e_head = only_train_e2e_head
        embed_dims_2 = 1024
        self.query_pos = nn.Sequential(
            nn.Linear(396, embed_dims_2),
            nn.ReLU(),
            nn.Linear(embed_dims_2, embed_dims),
        )
        if pts_bbox_head is None and map_head is None:
            for p in self.query_pos.parameters():
                p.requires_grad = False
        
        if not use_qwen_visual:
            # self.img_proj = nn.Conv2d(1024, embed_dims, 2, stride=2)
            down_sample_ratio = stride // 16
            self.down_sample_ratio = down_sample_ratio
            self.img_proj = nn.Sequential(
                nn.Conv2d(1024, embed_dims, down_sample_ratio, stride=down_sample_ratio),
                nn.ReLU(),
                nn.Conv2d(embed_dims, embed_dims, 1, stride=1),
            )

        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims)
        )

        self.ego_pose_pe = MLN(156, f_dim=embed_dims)
        self.use_mapemb = use_mapemb
        self.embed_dims = embed_dims
        self.img_num = img_num

        # TODO: same query? 
        # For box head, 这里是指在初始化一样，还是一直都是一个？
        if pts_bbox_head is not None:
            if use_same_query:
                self.pts_bbox_head.query_pos = self.query_pos
                self.pts_bbox_head.time_embedding = self.time_embedding
                self.pts_bbox_head.ego_pose_pe = self.ego_pose_pe

        if img_head is not None:
            self.img_head = builder.build_head(img_head)

        if map_head is not None:
            # TODO: same query?
            # For map head
            self.map_head = builder.build_head(map_head)
            if use_same_query:
                self.map_head.query_pos = self.query_pos
                self.map_head.time_embedding = self.time_embedding
                self.map_head.ego_pose_pe = self.ego_pose_pe

        if ego_status_embed is not None:
            self.ego_status_embed = builder.build_head(ego_status_embed)
            # if e2e_head is None: # freeze ego status if no e2e head
            for p in self.ego_status_embed.parameters():
                p.requires_grad = False


        if tokenizer is not None:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                        model_max_length=tokenizer_max_length,
                                        padding_side="right",
                                        use_fast=False,
                                        )
            if 'llava' in tokenizer:
                import ipdb; ipdb.set_trace()
                self.tokenizer.pad_token = self.tokenizer.unk_token

        else:
            self.tokenizer = None
        
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        if LID:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - depth_start) / (depth_num * (1 + depth_num))
            coords_d = depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            bin_size = (self.position_range[3] - depth_start) / depth_num
            coords_d = depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self.position_encoder = nn.Sequential(
                nn.Linear(depth_num*3, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
            )
        
        if e2e_head is not None:
            self.e2e_head = builder.build_head(e2e_head)
        
        if not use_qwen_visual and frozen_vit:
            for name, param in self.named_parameters():
                if 'img_backbone' in name or 'img_proj' in name:
                    param.requires_grad = False
        
        if lm_head is not None:
            # self.lm_head = load_model(lm_head, use_lora, frozen, finetune_emb)
            self.processor = AutoProcessor.from_pretrained(tokenizer, trust_remote_code=True)
            # self.lm_head = load_uni_model(lm_head, use_lora, frozen, finetune_emb,r=lora_rank, frozen_vit=frozen_vit)
            if pts_bbox_head is None and map_head is None:
                add_new_modules = False
                for n in self.time_embedding.parameters():
                    n.requires_grad = False
                for n in self.ego_pose_pe.parameters():
                    n.requires_grad = False
                
                if e2e_head is not None:
                    add_new_modules = True
            else:
                add_new_modules = True
            
            
            # TODO: merge load_uni_model and visual/task_adapter/... modification together
            self.use_layers = use_layers
            self.lm_head = load_uni_model(
                lm_head, use_lora, frozen, finetune_emb,
                r=lora_rank, frozen_vit=frozen_vit, 
                # add_new_modules=not full_ft,
                add_new_modules=add_new_modules,
                new_modules_list=['query_self_attn', 'query_mlp', 'query_gate', 'query_norm'], # only for unfrozen modules
                full_ft=full_ft,
                full_ft_attn=full_ft_attn,
                random_init=random_init,
                small_ffn=small_ffn,
                random_init_list=random_init_list,
                # only_e2e_head=only_e2e_head,
                use_e2e_head=self.with_e2e_head,
                use_det_head=self.with_pts_bbox_head,
                use_map_head=self.with_map_head,
                same_self_attn=same_self_attn,
                e2e_with_self_attn=e2e_with_self_attn,
                rm_temp_in_casual=rm_temp_in_casual,
                use_layers=use_layers,
            )

            

            for name, param in self.lm_head.named_parameters():
                if 'visual' in name:
                    param.requires_grad = not frozen_vit
                    # if not frozen_vit:
                    param.data = param.data.to(torch.float32)

                if 'task_adapter' in name:
                    param.requires_grad = True
                    param.data = param.data.to(torch.float32)
                    
                if 'task_self_attn' in name:
                    param.requires_grad = True
                    param.data = param.data.to(torch.float32)
                
                if 'task_norm' in name:
                    param.requires_grad = True
                    param.data = param.data.to(torch.float32)
            
            if llm_bf16:
                for param in filter(lambda p: p.requires_grad, self.lm_head.parameters()):
                    param.data = param.data.to(torch.bfloat16)

            
        self.test_flag = False
        # Add counters for FPS calculation
        self.total_time = 0
        self.num_frames = 0
        self.text_ft = text_finetune
        self.text_loss_lambda = text_loss_lambda
        
        if not text_finetune:
            if use_lora:
                self.lm_head.base_model.model.model.layers = self.lm_head.base_model.model.model.layers[:6]
                self.lm_head.base_model.model.model.lm_head = None
            elif full_ft:
                self.lm_head.model.layers = self.lm_head.model.layers[:6]
                self.lm_head.model.lm_head = None   
            else:
                raise NotImplementedError
        
        if self.only_train_e2e_head:
            self.freeze_qfromer()

        # Not use for Qwen2.5-VL. 

        # self.img_pos_down_sample = nn.Sequential(
        #     nn.Conv2d(256, 4096, kernel_size=2, stride=2)
        # )
        
    def freeze_qfromer(self):
        # raise NotImplementedError
        for name, param in self.named_parameters():
            if 'e2e_head' not in name:
                param.requires_grad = False
        
        for name, param in self.named_parameters():
            if 'query_mlp_e2e' in name:
                param.requires_grad = True
            if 'task_self_attn_e2e' in name:
                param.requires_grad = True  
                
        train_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"需要梯度的参数: {name}")
                train_params.append(param)
        if len(train_params) == 0:
            raise ValueError("没有需要训练的参数")


    @property
    def with_map_head(self):
        """bool: Whether the detector has a map head."""
        return hasattr(self,
                       'map_head') and self.map_head is not None
    
    @property
    def with_img_head(self):
        """bool: Whether the detector has a img head."""
        return hasattr(self,
                       'img_head') and self.img_head is not None
        
    @property
    def with_pts_bbox_head(self):
        """bool: Whether the detector has a pts bbox head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None
    @property
    def with_lm_head(self):
        """bool: Whether the detector has a lm head."""
        return hasattr(self,
                       'lm_head') and self.lm_head is not None
    
    @property
    def with_e2e_head(self):
        """bool: Whether the detector has a e2e head."""
        return hasattr(self,
                       'e2e_head') and self.e2e_head is not None
        

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()

        img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B), C, H, W)


        return img_feats_reshaped


    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        return img_feats


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    def position_embeding(self, data, memory_centers, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        # memory_centers: pixel coordinate
        

        D = self.coords_d.shape[0] # depth num

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = memory_centers.repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)

        coords_position_embeding = self.position_encoder(pos_embed)

        return coords_position_embeding


    def get_pixel_values(self, img):
        # refer to qwen2-vl img processor
        batch_size, num_camera, channels = img.shape[:3]
        img = img.reshape(-1, *img.shape[2:])  # [b*n, c, h, w]
        img = img.unsqueeze(1) # [b*n, 1, c, h, w]
        # repeat to align with temporal size of Qwen2.5-VL
        img = img.repeat(1, 2, 1, 1, 1) # [b*n, 2, c, h, w] 
        
        # ref Qwen processor
        temp_patch_size = 2
        patch_size = 14
        merge_size = 2
        t, h, w = img.shape[1] // temp_patch_size, img.shape[3] // patch_size, img.shape[4] // patch_size
        batch_thw = [[t, h, w] for _ in range(batch_size * num_camera)]
        batch_thw = torch.tensor(batch_thw).to(img.device) # [b*n, 3]
        batch_img_patches = img.reshape(
            batch_size * num_camera, # 0
            t,  # 1 
            temp_patch_size, # 2
            channels, # c = 3 # 3
            h // merge_size, merge_size, patch_size, # 4, 5, 6
            w // merge_size, merge_size, patch_size # 7, 8, 9
        )

        batch_img_patches = batch_img_patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
        batch_img_patches = batch_img_patches.reshape(
            batch_size * num_camera, # 0
            t * h * w, 
            channels * temp_patch_size * patch_size * patch_size # c = 3 # 1, 2
        ).to(self.lm_head.visual.dtype)
        
        # 使用batch_img_patches，直接输入给Qwen2.5-VL的generate函数，能够给出正常结果。
        
        return batch_img_patches, batch_thw
    
    def get_img_feats_by_QwenVisual_single_frame(self, img):
        # img.shape = [b, n, c, h, w]
        assert self.lm_head.visual is not None
        batch_size, num_camera, channels = img.shape[:3]
        img = img.reshape(-1, *img.shape[2:])  # [b*n, c, h, w]
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img = img.unsqueeze(1) # [b*n, 1, c, h, w]
        # repeat to align with temporal size of Qwen2.5-VL
        img = img.repeat(1, 2, 1, 1, 1) # [b*n, 2, c, h, w] 
        
        # ref Qwen processor
        temp_patch_size = 2
        patch_size = 14
        merge_size = 2
        t, h, w = img.shape[1] // temp_patch_size, img.shape[3] // patch_size, img.shape[4] // patch_size
        batch_thw = [[t, h, w] for _ in range(batch_size * num_camera)]
        batch_thw = torch.tensor(batch_thw).to(img.device) # [b*n, 3]
        batch_img_patches = img.reshape(
            batch_size * num_camera, # 0
            t,  # 1 
            temp_patch_size, # 2
            channels, # c = 3 # 3
            h // merge_size, merge_size, patch_size, # 4, 5, 6
            w // merge_size, merge_size, patch_size # 7, 8, 9
        )

        batch_img_patches = batch_img_patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
        batch_img_patches = batch_img_patches.reshape(
            batch_size * num_camera, # 0
            t * h * w, 
            channels * temp_patch_size * patch_size * patch_size # c = 3 # 1, 2
        ).to(self.lm_head.visual.dtype)

        img_embeddings = self.lm_head.visual(batch_img_patches, batch_thw)  # [b*n, t*h*w, c]
        img_embeddings = img_embeddings.view(batch_size, -1, self.embed_dims)
        
        return img_embeddings, batch_thw

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          input_ids, 
                          vlm_labels, 
                          vlm_attn_mask,
                          lane_pts,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        if len(data['img'].shape) == 4: # means no batch dim, add batch dim
            data['img'] = data['img'].unsqueeze(0) 
        B, N_CAM = data['img'].shape[:2]

        if N_CAM > self.img_num:
            N_CAM = self.img_num
            data['img'] = data['img'][:, :N_CAM, ...]
            if 'img_feats' in data.keys():
                data['img_feats'] = data['img_feats'][:, :N_CAM, ...]
            data['intrinsics'] = data['intrinsics'][:, :N_CAM, ...]
            data['lidar2img'] = data['lidar2img'][:, :N_CAM, ...]

        # begin = time.time()
        # with torch.no_grad():
        if self.use_qwen_visual:
            img_embedding, batch_thw = self.get_img_feats_by_QwenVisual_single_frame(data['img'])
            # hack
            h, w = batch_thw[0][1:] / 2
            data['img_feats'] = img_embedding.view(B, N_CAM, round(h.item()), round(w.item()), self.embed_dims).permute(0, 1, 4, 2, 3).contiguous() 
        else:
            # using EVA backbone
            # # hack imple
            # t, h, w = 1, 40, 40 # hack
            t, h, w = 1, 20, 20 # hack
            batch_thw = [[t, h, w] for _ in range(B * N_CAM)]
            batch_thw = torch.tensor(batch_thw).to(data['img'].device) # [b*n, 3]
            img_embedding = self.img_proj(data['img_feats'].flatten(0,1)).reshape(B, N_CAM, self.embed_dims, h, w)
            img_embedding = img_embedding.permute(0, 1, 3, 4, 2).contiguous().view(B, N_CAM*h*w, self.embed_dims)

        # prepare img pos embedding
        location = self.prepare_location(img_metas, **data)
        pos_embeds_3d = self.position_embeding(data, location, img_metas)
        
        # print("Get img embedding time:", time.time() - begin)
        img_len = img_embedding.shape[1]

        # Prepare task tokens: get all tasks query embedding
        query_embedding = []
        query_pos = []
        query_lens = []
        ref_lens = []
        query_begins = [0]
        reference_dict = {}

        # begin = time.time()
        # can_bus, can_bus_embedding = self.ego_status_embed.get_query_embedding(B, data)
        used_heads= []
        if self.with_pts_bbox_head:
            used_heads.append('pts_bbox_head')
        if self.with_map_head:
            used_heads.append('map_head')
        if self.with_e2e_head:
            used_heads.append('e2e_head')
            
        for head_name in used_heads:
            if hasattr(self, head_name) and getattr(self, head_name) is not None:
                head = getattr(self, head_name)
                query_embed, pos_embed, reference = head.get_query_embedding(B, data)
                if head_name == 'e2e_head':
                    data['plan_embeds_before_llm'] = query_embed[:, 1:]
                    data['can_bus_embeds_before_llm'] = query_embed[:, :1]
                query_embedding.append(query_embed)
                query_pos.append(pos_embed)
                query_lens.append(query_embed.shape[1])
                query_begins.append(query_begins[-1] + query_embed.shape[1])
                reference_dict[head_name] = reference # can be None
                ref_lens.append(reference.shape[1] if head_name!='e2e_head' else query_embed.shape[1])
        
        if len(used_heads) == 0:
            query_pos = query_embedding = None
            query_len = 0
        else:
            query_embedding = torch.cat(query_embedding, dim=1)
            query_pos = torch.cat(query_pos, dim=1)
            query_len = query_embedding.shape[1]
        
        tasks_metas = {
            'used_heads': used_heads,
            'query_lens': query_lens,
            'query_begins': query_begins[:-1], # remove last one
            'ref_lens': ref_lens,
        }
        
        losses = dict()

        # begin = time.time()
        vlm_out = self.lm_head(
            input_ids=input_ids,
            attention_mask=vlm_attn_mask,
            tasks_embeds=query_embedding,
            tasks_pos_embeds=query_pos,
            tasks_metas=tasks_metas,
            images_embeds=img_embedding,
            images_pos_embeds_3d=pos_embeds_3d,
            images_thw=batch_thw,
            output_hidden_states=True,
            labels=vlm_labels,
            use_cache=False
        )

        # print("Forward LLM time:", time.time() - begin)
        # begin = time.time()

        # dense supervision for task queries    
        if len(used_heads) > 0:
            # group_layer_depth = 3 # hack implementation
            if 'task_hidden_states' in vlm_out:
                query_embedding = torch.stack(vlm_out['task_hidden_states'], 0).to(torch.float32)
            else:
                raise NotImplementedError   
            # else:
            #     group_layer_depth = 1 # hack implementation
            #     query_embedding = torch.stack(vlm_out['hidden_states'][group_layer_depth::group_layer_depth], 0)
            #     # query_embedding = vlm_out['hidden_states'][-1]
                
            #     query_with_img_mask = vlm_out['bidirectional_mask'].unsqueeze(0).repeat(query_embedding.shape[0], 1, 1)
            #     query_embedding = query_embedding[query_with_img_mask].reshape(-1, B, query_len + img_len, self.embed_dims)[:, :, -query_len:, :]
            #     query_embedding = query_embedding.to(torch.float32)
        
        if self.text_ft:
            vlm_loss = vlm_out['loss']
            vlm_loss = torch.nan_to_num(vlm_loss)
            losses.update(vlm_loss=vlm_loss * self.text_loss_lambda)

        
        # forward heads
        tasks_outs = {}
        for head_name in used_heads: 
            if hasattr(self, head_name) and getattr(self, head_name) is not None:
                head = getattr(self, head_name)
                beg = query_begins[used_heads.index(head_name)]
                end = beg + query_lens[used_heads.index(head_name)]
                _query = query_embedding[..., beg:end, :]
                
                reference = reference_dict[head_name]   

                # if head_name == 'e2e_head':
                #     # hack implementation: use last mixed attention layers
                #     _query = _query[5]
                
                _tasks_outs = head(
                    _query, 
                    # can_bus=can_bus, 
                    reference_points=reference, 
                    img_metas=img_metas,
                    data=data,    
                )
                tasks_outs[head_name] = _tasks_outs
                
                if head_name == 'pts_bbox_head':
                    loss_inputs = [gt_bboxes_3d, gt_labels_3d, _tasks_outs]
                    loss = head.loss(*loss_inputs)
                    losses.update(loss)
                elif head_name == 'map_head':
                    loss_inputs = [lane_pts, _tasks_outs, img_metas]
                    loss = head.loss(*loss_inputs)
                    losses.update(loss)
                elif head_name == 'e2e_head':
                    losses.update(_tasks_outs)
                else:
                    raise NotImplementedError
        # print("Total heads time:", time.time() - begin)
        return losses

    # @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      input_ids=None,
                      vlm_labels=None,
                      lane_pts=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            # self.test_flag = False
        if self.tokenizer is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            
            vlm_labels = torch.nn.utils.rnn.pad_sequence(vlm_labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            vlm_labels = vlm_labels[:, :self.tokenizer.model_max_length]
            vlm_attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            input_ids = None
            vlm_labels = None
            vlm_attn_mask = None
        if not self.use_qwen_visual:
            data['img_feats'] = self.extract_feat(data['img'])

        losses = self.forward_pts_train(gt_bboxes_3d,
                                    gt_labels_3d, gt_bboxes,
                                    gt_labels, img_metas, centers2d, 
                                    depths, input_ids, vlm_labels, vlm_attn_mask, lane_pts, **data)

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        if not self.test_flag: #for interval evaluation
            if self.with_pts_bbox:
                self.pts_bbox_head.reset_memory()
            if self.with_map_head:
                self.map_head.reset_memory()
            self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key not in ['img', 'input_ids']:
                try:
                    data[key] = data[key][0][0].unsqueeze(0)
                except:
                    pass  # hack implementation for loading annotation like forward_train
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        # timing = {}
        # torch.cuda.synchronize()
        # total_start = time.time()
        if len(data['img'].shape) == 4: # means no batch dim, add batch dim
            data['img'] = data['img'].unsqueeze(0) 
        B, N_CAM = data['img'].shape[:2]
        # get image features 
        if self.use_qwen_visual:
            img_embedding, batch_thw = self.get_img_feats_by_QwenVisual_single_frame(data['img'])
            h, w = batch_thw[0][1:] / 2
            data['img_feats'] = img_embedding.view(B, N_CAM, round(h.item()), round(w.item()), self.embed_dims).permute(0, 1, 4, 2, 3).contiguous()
        else:
            raise NotImplementedError

        # prepare img pos embedding
        location = self.prepare_location(img_metas, **data)
        pos_embeds_3d = self.position_embeding(data, location, img_metas)

        img_len = img_embedding.shape[1]
        
        # parepare tasks tokens
        # query_embedding, query_pos, query_lens, query_begins, reference_dict = [], [], [], [0], {}
        query_embedding = []
        query_pos = []
        query_lens = []
        ref_lens = []
        query_begins = [0]
        reference_dict = {}

        
        bbox_results = []

        used_heads = []
        if self.with_pts_bbox_head:
            used_heads.append('pts_bbox_head')
        if self.with_map_head:
            used_heads.append('map_head')
        if self.with_e2e_head:
            used_heads.append('e2e_head')
            
        for head_name in used_heads:
            if hasattr(self, head_name) and getattr(self, head_name) is not None:
                head = getattr(self, head_name)
                query_embed, pos_embed, reference = head.get_query_embedding(B, data)
                if head_name == 'e2e_head':
                    data['plan_embeds_before_llm'] = query_embed[:, 1:]
                    data['can_bus_embeds_before_llm'] = query_embed[:, :1]
                query_embedding.append(query_embed)
                query_pos.append(pos_embed)
                query_lens.append(query_embed.shape[1])
                query_begins.append(query_begins[-1] + query_embed.shape[1])
                reference_dict[head_name] = reference
                ref_lens.append(reference.shape[1] if head_name!='e2e_head' else query_embed.shape[1])
                
        if len(used_heads) == 0:
            query_pos = query_embedding = None
            query_len = 0
        else:
            query_embedding = torch.cat(query_embedding, dim=1)
            query_pos = torch.cat(query_pos, dim=1)
            query_len = query_embedding.shape[1]
        
        tasks_metas = {
            'used_heads': used_heads,
            'query_lens': query_lens,
            'query_begins': query_begins[:-1], # remove last one
            'ref_lens': ref_lens,
        }
        
        # Language model head        
        generated_text = []
        # -200 img token; -202 point token
        vlm_out = self.lm_head.generate(
            input_ids=data['input_ids'][0][0].unsqueeze(0),
            tasks_embeds=query_embedding,
            tasks_pos_embeds=query_pos,
            tasks_metas=tasks_metas,
            images_embeds=img_embedding,
            images_pos_embeds_3d=pos_embeds_3d,
            images_thw=batch_thw,
            return_dict_in_generate=True,
            output_hidden_states=True,
            # max_new_tokens=320,
            max_new_tokens=1, # only evaluate query embedding
            # use_cache=False # 
        )
        
        
        tasks_res = {}  
        bbox_results = []
        lane_results = []
        if len(used_heads) > 0:
            if 'task_hidden_states' in vlm_out:
                out_query_embedding = torch.stack(vlm_out['task_hidden_states'], 0).to(torch.float32)
            else:
                # bidirectional_mask = data['input_ids'][0][0].unsqueeze(0) == self.lm_head.config.image_token_id
                # hidden_states = vlm_out['hidden_states'][0][18] # [i][j]: i-th generate step, j-th layer
                # out_query_embedding = hidden_states[bidirectional_mask].reshape(B, query_len + img_len, -1)[:, :query_len, :]

                # group_layer_depth = 3 # hack implementation
                group_layer_depth = 1 # hack implementation
                out_query_embedding = torch.stack(vlm_out['hidden_states'][0][group_layer_depth::group_layer_depth], 0)

                # hack imple, img pos should fixed
                bidirectional_mask = data['input_ids'][0][0] == self.lm_head.config.image_token_id
                out_query_embedding = out_query_embedding[:, :, bidirectional_mask].reshape(-1, B, query_len + img_len, self.embed_dims)[:, :, -query_len:, :]
                out_query_embedding = out_query_embedding.to(torch.float32) 

            for head_name in used_heads:
                if hasattr(self, head_name) and getattr(self, head_name) is not None:
                    head = getattr(self, head_name)
                    beg = query_begins[used_heads.index(head_name)]
                    end = beg + query_lens[used_heads.index(head_name)]
                    output_layer_id = self.use_layers[-1]
                    _query = out_query_embedding[output_layer_id, :,  beg:end, :]                
                    reference = reference_dict[head_name]   

                    if head_name == 'pts_bbox_head' or head_name == 'map_head':
                        _query = _query.unsqueeze(0)

                    _tasks_outs = head(
                        _query, 
                        reference_points=reference, 
                        img_metas=img_metas,
                        data=data,    
                    )
                    
                    if head_name == 'pts_bbox_head':
                        bbox_list = head.get_bboxes(_tasks_outs, img_metas)
                        for bboxes, scores, labels in bbox_list:
                            bbox_results.append(bbox3d2result(bboxes, scores, labels))
                            
                    elif head_name == 'map_head':
                        lane_results = head.get_bboxes(_tasks_outs, img_metas)
                    
                    tasks_res[head_name] = _tasks_outs

        generated_text = []
        for i, input_ids in enumerate(data['input_ids'][0]):
            break
            input_ids = input_ids.unsqueeze(0)
            vlm_out = self.lm_head.generate(
                input_ids=input_ids,
                tasks_embeds=query_embedding,
                tasks_pos_embeds=query_pos,
                images_embeds=img_embedding,
                images_pos_embeds_3d=pos_embeds_3d,
                images_thw=batch_thw,
                return_dict_in_generate=True,
                max_new_tokens=320,
                # output_hidden_states=True,
                # use_cache=False # 
            )
            output_ids = vlm_out['sequences']
            generated_text.append(
                # tokenizer inputs > 0
                dict(
                    Q=img_metas[0]['vlm_labels'].data[0],
                    A=self.tokenizer.batch_decode(output_ids, skip_special_tokens=True),
                )
            )
            break
            # expect first question, other questions without Image
        
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        
        if not os.path.exists(os.path.join(self.save_path, 'text_results')):
            os.makedirs(os.path.join(self.save_path, 'text_results'), exist_ok=True)
        
        # ? .pkl
        with open(os.path.join(self.save_path, 'text_results', img_metas[0]['sample_idx']), 'w') as file:
            json.dump(generated_text, file)
    
        if self.with_e2e_head:
            plan_reg_pred = tasks_res['e2e_head']
            # if not os.path.exists(os.path.join(self.save_path, 'e2e_results_v12_v2head_gt')):
            #     os.makedirs(os.path.join(self.save_path, 'e2e_results_v12_v2head_gt'), exist_ok=True)
            # with open(os.path.join(self.save_path, 'e2e_results_v12_v2head_gt', img_metas[0]['sample_idx'] + '.pkl'), 'wb') as file:
            #     pickle.dump((plan_reg_pred.cpu().numpy()), file)
            
            if not os.path.exists(os.path.join(self.save_path, 'e2e_results')):
                os.makedirs(os.path.join(self.save_path, 'e2e_results'), exist_ok=True)
            with open(os.path.join(self.save_path, 'e2e_results', img_metas[0]['sample_idx'] + '.pkl'), 'wb') as file:
                pickle.dump((plan_reg_pred.cpu().numpy()), file)

        return bbox_results, generated_text, lane_results
        
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        torch.cuda.synchronize()
        t_start = time.time()
        
        # data['img_feats'] = self.extract_img_feat(data['img'])
        
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts, generated_text, lane_results = self.simple_test_pts(
            img_metas, **data)
        
        torch.cuda.synchronize()
        frame_time = time.time() - t_start
        
        # Update running statistics
        self.num_frames += 1
        if self.num_frames > 100:  # Only count after warmup
            self.total_time += frame_time
            avg_fps = (self.num_frames - 100) / self.total_time
            # print('avg_fps', avg_fps)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        bbox_list[0]['text_out'] = generated_text
        bbox_list[0]['lane_results'] = lane_results
        return bbox_list
    

    