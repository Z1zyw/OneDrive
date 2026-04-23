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
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points_lane as transform_reference_points, memory_refresh, SELayer_Linear

from mmcv.cnn import xavier_init

import numpy as np
import os

@HEADS.register_module()
class UniMapHeadSeq(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 out_dims=4096,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_extra=256,
                 n_control=11,
                 use_learnable_query=False,
                 can_bus_len=2,
                 pc_range=None,
                 with_mask=False,
                 with_dn=False,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                #  transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None, # can not be none 
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 normedlinear=False,
                 use_commandemb=False,
                 save_vlm_memory=False,
                 save_path='./',
                 save_path_canbus='./',
                 only_cat_query=False,
                 cat_memory=False,
                 num_pred=12, # output layers, refine layer by layer
                 input_norm=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        self.use_learnable_query = use_learnable_query
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            # self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0., 0.]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is UniMapHeadSeq):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.output_dims = out_dims
        self.n_control = n_control
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_extra = num_extra
        assert not with_dn, "dn_train is not support currently"
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.with_mask = with_mask
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.embed_dims = embed_dims
        self.save_vlm_memory = save_vlm_memory
        self.save_path = save_path
        self.save_path_canbus = save_path_canbus
        self.only_cat_query = only_cat_query
        self.cat_memory = cat_memory
        
        self.num_lane = num_query

        # self.act_cfg = transformer.get('act_cfg',
                                    #    dict(type='ReLU', inplace=True))
        self.num_pred = num_pred
        self.input_norm = input_norm
        self.normedlinear = normedlinear
        super(UniMapHeadSeq, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dir = build_loss(loss_dir)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        # self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.pc_range = nn.Parameter(torch.tensor(
            pc_range), requires_grad=False)
        # self.bbox_coder = build_bbox_coder(bbox_coder)

        # self.pc_range = nn.Parameter(torch.tensor(
        #     self.bbox_coder.pc_range), requires_grad=False)

        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.input_norm == True:
            _input_norm = nn.LayerNorm(self.embed_dims)
            self.input_norm = nn.ModuleList([_input_norm for _ in range(self.num_pred)])
        else:
            _input_norm = nn.Identity()
            self.input_norm = nn.ModuleList([_input_norm for _ in range(self.num_pred)])
        
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.n_control*3))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])
        
        self.reference_points_lane = nn.Linear(self.embed_dims, 3)
        self.points_embedding_lane = nn.Embedding(self.n_control, self.embed_dims)
        self.instance_embedding_lane = nn.Embedding(self.num_lane, self.embed_dims)
        
        self.query_pos = None
        self.time_embedding = None
        self.ego_pose_pe = None

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        # self.transformer.init_weights()
        xavier_init(self.reference_points_lane, distribution='uniform', bias=0.)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
         
    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.sample_time = None

    def pre_update_memory(self, data):
        B = data['img_feats'].size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = data['img_feats'].new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = data['img_feats'].new_zeros(B, self.memory_len, self.n_control, 3)
            self.memory_timestamp = data['img_feats'].new_zeros(B, self.memory_len, 1)
            self.memory_egopose = data['img_feats'].new_zeros(B, self.memory_len, 4, 4)
            self.sample_time = data['timestamp'].new_zeros(B)
            x = self.sample_time.to(data['img_feats'].dtype)
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.sample_time += data['timestamp']
            x = (torch.abs(self.sample_time) < 2.0).to(data['img_feats'].dtype)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.sample_time = data['timestamp'].new_zeros(B)

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec):

        rec_reference_points = all_bbox_preds[-1].reshape(outs_dec.shape[1], -1, self.n_control, 3)
        out_memory = outs_dec[-1]
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(out_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        
        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.sample_time -= data['timestamp']
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
        
        return out_memory

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_pos(nerf_positional_encoding(temp_reference_point.flatten(-2))) 
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1]),num_pos_feats=self.embed_dims).float())
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp, num_pos_feats=self.embed_dims).float())

        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is UniMapHeadSeq:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def get_query_embedding(self, batch_size:int, data):
        self.pre_update_memory(data)
        
        lane_embedding = self.instance_embedding_lane.weight.unsqueeze(-2) + self.points_embedding_lane.weight.unsqueeze(0) 
        reference_points_lane = self.reference_points_lane(lane_embedding).sigmoid().flatten(-2).unsqueeze(0).repeat(batch_size, 1, 1)
        query_pos = self.query_pos(nerf_positional_encoding(reference_points_lane))
        
        if self.use_learnable_query:
            query_embed = self.instance_embedding_lane.weight.unsqueeze(0).repeat(batch_size, 1, 1) # b, 1800. 256
        else:
            query_embed = torch.zeros_like(query_pos)
        
        query_embed, query_pos, reference_points_lane, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, query_embed, reference_points_lane)

        # Maybe BUG here: self.rec_ego_pose
        self.rec_ego_pose = rec_ego_pose
        query_embed = torch.cat([query_embed, temp_memory], dim=1)
        query_pos = torch.cat([query_pos, temp_pos], dim=1)
        
        return query_embed, query_pos, reference_points_lane
        
    def forward(
        self, 
        query, 
        reference_points,
        # img_ref, 
        img_metas, 
        data,
        **kwargs):
        outs_dec = query.clone()
        
        outputs_classes = []
        outputs_coords = []
        
        output_len = reference_points.shape[1]
        outs_dec = outs_dec[:, :, :output_len, :]

        if self.training:

            assert outs_dec.shape[0] >= self.num_pred
            for lvl in range(self.num_pred):
                out_dec = self.input_norm[lvl](outs_dec[lvl])
                reference = inverse_sigmoid(reference_points.clone())
                reference = reference.view(-1, self.num_lane, self.n_control*3)
                tmp = self.reg_branches[lvl](out_dec)
                outputs_lanecls = self.cls_branches[lvl](out_dec)

                tmp = tmp.reshape(-1, self.num_lane, self.n_control*3)
                tmp += reference
                tmp = tmp.sigmoid()

                outputs_coord = tmp
                outputs_coord = outputs_coord.reshape(-1, self.num_lane, self.n_control, 3)
                outputs_coords.append(outputs_coord)
                outputs_classes.append(outputs_lanecls)
                
            # for lvl in range(self.num_pred):
            #     reference = inverse_sigmoid(reference_points.clone())
            #     assert reference.shape[-1] == 3
            #     outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            #     tmp = self.reg_branches[lvl](outs_dec[lvl])

            #     tmp[..., 0:3] += reference[..., 0:3]
            #     tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            #     outputs_coord = tmp
            #     outputs_classes.append(outputs_class)
            #     outputs_coords.append(outputs_coord)
        else:
            # only input the last decoder layer when testing
            out_dec = self.input_norm[-1](outs_dec[-1])
            
            reference = inverse_sigmoid(reference_points.clone())
            reference = reference.view(-1, self.num_lane, self.n_control*3)
            # assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[-1](out_dec)
            tmp = self.reg_branches[-1](out_dec)

            tmp = tmp.reshape(-1, self.num_lane, self.n_control*3)
            tmp += reference
            tmp = tmp.sigmoid()
            # tmp[..., 0:3] += reference[..., 0:3]
            # tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp.reshape(-1, self.num_lane, self.n_control, 3)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        all_bbox_preds = all_bbox_preds.flatten(-2)
        

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
        }

        _ = self.post_update_memory(data, self.rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec)
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, img_meta, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)

        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, img_metas,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        dir_weights = bbox_weights.reshape(-1, self.n_control, 3)[:, :-1,0]
        pts_preds_dir = bbox_preds.reshape(-1, self.n_control, 3)[:,1:,:] - bbox_preds.reshape(-1, self.n_control, 3)[:,:-1,:]
        pts_targets_dir = bbox_targets.reshape(-1, self.n_control, 3)[:, 1:,:] - bbox_targets.reshape(-1, self.n_control, 3)[:,:-1,:]
        loss_dir = self.loss_dir(
            pts_preds_dir, pts_targets_dir,
            dir_weights,
            avg_factor=num_total_pos)
        bbox_preds = bbox_preds.reshape(-1, self.n_control * 3)
        
        # bbox_preds = self.control_points_to_lane_points(bbox_preds)
        # bbox_targets = self.control_points_to_lane_points(bbox_targets)
        bbox_weights = bbox_weights.mean(-1).unsqueeze(-1).repeat(1, bbox_preds.shape[-1])
        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        return loss_cls, loss_bbox, loss_dir
    
    # @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
            #  gt_bboxes_list,
            #  gt_labels_list,
             gt_lanes,
             preds_dicts,
             img_metas=None,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        # device = gt_labels_list[0].device
        gt_lanes = [lane.reshape(-1, self.n_control*3) for lane in gt_lanes]
        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        gt_labels = [torch.zeros(gt_lane.shape[0], dtype=torch.long, device=gt_lanes[0].device) for gt_lane in gt_lanes]
        all_gt_labels_list = [gt_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_list = [gt_lanes for _ in range(num_dec_layers)]
        # all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_bbox, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls_lane'] = losses_cls[-1]
        loss_dict['loss_bbox_lane'] = losses_bbox[-1]
        # loss_dict['loss_dir'] = loss_dir

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_dir_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1], losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_lane'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_lane'] = loss_bbox_i
            # loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1

        return loss_dict


    # # @force_fp32(apply_to=('preds_dicts'))
    # def get_bboxes(self, preds_dicts, img_metas, rescale=False):
    #     """Generate bboxes from bbox head predictions.
    #     Args:
    #         preds_dicts (tuple[list[dict]]): Prediction results.
    #         img_metas (list[dict]): Point cloud and image's meta info.
    #     Returns:
    #         list[dict]: Decoded bbox, scores and labels after nms.
    #     """
    #     preds_dicts = self.bbox_coder.decode(preds_dicts)
    #     num_samples = len(preds_dicts)

    #     ret_list = []
    #     for i in range(num_samples):
    #         preds = preds_dicts[i]
    #         bboxes = preds['bboxes']
    #         bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
    #         bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
    #         scores = preds['scores']
    #         labels = preds['labels']
    #         ret_list.append([bboxes, scores, labels])
    #     return ret_list
    
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        cls_scores = preds_dicts['all_cls_scores'][-1]
        bbox_preds = preds_dicts['all_bbox_preds'][-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            boxes, scores = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append([boxes, scores])
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):

        assert len(cls_score) == len(bbox_pred)
        cls_score = cls_score.sigmoid()
        det_bboxes = bbox_pred
        for p in range(self.n_control):
            det_bboxes[..., 3 * p].clamp_(min=self.pc_range[0], max=self.pc_range[3])
            det_bboxes[..., 3 * p + 1].clamp_(min=self.pc_range[1], max=self.pc_range[4])
            
        # det_bboxes = self.control_points_to_lane_points(det_bboxes)
        det_bboxes = det_bboxes.reshape(det_bboxes.shape[0], -1, 3)

        return det_bboxes.cpu().numpy(), cls_score.cpu().numpy()