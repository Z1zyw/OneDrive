from tkinter import N
from typing import List, Optional, Tuple, Union
import warnings
import copy
import mmcv
import numpy as np
import cv2
import torch
import torch.nn as nn

# v6 使用 deepsupervision，每一层decoder 都会有 loss，这样会好吗

# from mmengine.registry import build_from_cfg
from mmcv.cnn import Linear
from mmdet.models import HEADS, build_loss
# from mmengine.model import bias_init_with_prob
# from mmengine.model import BaseModule
#force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
# from mmdet3d.registry import MODELS as HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

# from data_gen import planning_anchor
from ..utils.positional_encoding import pos2posemb2d, gen_sineembed_for_position
# from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
# from projects.mmdet3d_plugin.core.box3d import *

from ..utils.petr_transformer import ConstructiveTransformerLayer, PETRTransformerDecoderLayer

from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear


@HEADS.register_module()
class UniMotionPlanningHeadEgo(nn.Module):
    def __init__(
        self,
        ego_fut_ts=6,
        ego_fut_mode=3,
        embed_dims=256,
        embed_dims_2=256,
        using_gt_inference=False, # for ablation study
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        single_anchor=False,
        use_anchor_match=False, 
        kmeans_anchor_path='/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/data/kmeans/kmeans_plan_6.npy',
        pretrained_path=None,
        future_frame_num=2,
        future_traj_path_train=None,
        future_traj_path_val=None,
        num_extra=256,
        use_seq_query=False,
        two_layer_cross_attn=False,
        add_plan_before_cross_attn=False,
        canbus_wo_attn=False,
        add_attn_before_head=False,
        encode_all_traj=False,
        pred_res=False,
        num_pred=6, # deep supervision
        sync_pred_traj_path_train=None,
        sync_pred_traj_path_val=None,
        input_norm=False,
    ):
        super(UniMotionPlanningHeadEgo, self).__init__()


        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.num_extra = num_extra
        self.ego_fut_mode = ego_fut_mode
        self.pretrained_path = pretrained_path
        self.two_layer_cross_attn = two_layer_cross_attn
        self.add_plan_before_cross_attn = add_plan_before_cross_attn
        self.canbus_wo_attn = canbus_wo_attn
        
        self.using_gt_inference = using_gt_inference
        self.use_anchor_match = use_anchor_match
        
        
        self.can_bus_len = 2
        self.use_commandemb = False
        
        
        self.add_attn_before_head = add_attn_before_head
        self.encode_all_traj = encode_all_traj
        self.pred_res = pred_res
        self.sync_pred_traj = None
        self.sync_pred_traj_train = None
        self.sync_pred_traj_val = None
        if sync_pred_traj_path_train is not None:
            self.sync_pred_traj_train = mmcv.load(sync_pred_traj_path_train)
        if sync_pred_traj_path_val is not None:
            self.sync_pred_traj_val = mmcv.load(sync_pred_traj_path_val)
        if self.training:
            self.sync_pred_traj = self.sync_pred_traj_train
        else:
            self.sync_pred_traj = self.sync_pred_traj_val
        plan_anchor_lidar = torch.from_numpy(np.load(kmeans_anchor_path)).cuda().float()
        plan_anchor_ego = plan_anchor_lidar.clone()
        plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
        plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
        self.plan_anchor = plan_anchor_ego[[1,0,2]] # 0: left, 1: right, 2: forward
        if single_anchor:
            self.plan_anchor = self.plan_anchor.mean(dim=1, keepdim=True)
        self.plan_anchor.requires_grad = False
        
        # embed_dims_2 = embed_dims
        # # embed_dims_2 = 256
        
        self.temporal_embedding = nn.Embedding(6, embed_dims)
        
        self.mlp_position_encoder = nn.Sequential(
            nn.Linear(1, embed_dims_2),
            nn.ReLU(),
            nn.Linear(embed_dims_2, embed_dims),
        )
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)
        self.planning_sampler = PlanningTarget(
            ego_fut_ts=6, ego_fut_mode=self.plan_anchor.shape[1]+1  if sync_pred_traj_path_train is not None else self.plan_anchor.shape[1],
            use_anchor_match=use_anchor_match
        )
        self.use_seq_query = use_seq_query
        input_len = 74
        self.can_bus_embed = nn.Sequential(
            nn.Linear(input_len, embed_dims_2), # canbus + command + egopose
            nn.ReLU(),
            nn.Linear(embed_dims_2, embed_dims),)

        plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims_2),
            nn.ReLU(),
            nn.Linear(embed_dims_2, embed_dims_2),
            nn.ReLU(),
            nn.Linear(embed_dims_2, 2),
        )
        self.plan_reg_branches = nn.ModuleList([plan_reg_branch for _ in range(num_pred)])

        plan_cls_branch = nn.Sequential(
            # *linear_relu_ln(embed_dims, 1, 2),
            # nn.Linear(embed_dims, 1),)
            *linear_relu_ln(embed_dims_2, 1, 2, input_dims=embed_dims),
            nn.Linear(embed_dims_2, 1),)
        self.plan_cls_branches = nn.ModuleList([plan_cls_branch for _ in range(num_pred)])

        plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims_2),
            nn.ReLU(),
            nn.Linear(embed_dims_2, embed_dims_2),
            nn.ReLU(),
            nn.Linear(embed_dims_2, 74))
        self.plan_status_branches = nn.ModuleList([plan_status_branch for _ in range(num_pred)])
        
        if self.encode_all_traj:
            raise NotImplementedError
            self.plan_anchor_encoder = nn.Sequential(
                # *linear_relu_ln(6*embed_dims, 1, 1),
                # nn.Linear(6*embed_dims, embed_dims),
                *linear_relu_ln(6*embed_dims, 1, 1),
                nn.Linear(6*embed_dims, embed_dims),
            )
        else:   
            self.plan_anchor_encoder = nn.Sequential(
                # *linear_relu_ln(embed_dims, 1, 1),
                # nn.Linear(embed_dims, embed_dims),
                *linear_relu_ln(embed_dims_2, 1, 1, input_dims=embed_dims),
                nn.Linear(embed_dims_2, embed_dims),
            )

        self.future_frame_num = future_frame_num
        if future_traj_path_train is not None:  
            self.future_traj = mmcv.load(future_traj_path_train)
            self.future_traj_val = mmcv.load(future_traj_path_val)
        else:
            self.future_traj = None
            self.future_traj_val = None
        
        if input_norm:
            _input_norm = nn.LayerNorm(self.embed_dims)
            self.input_norm = nn.ModuleList([_input_norm for _ in range(num_pred)])
        else:
            _input_norm = nn.Identity()
            self.input_norm = nn.ModuleList([_input_norm for _ in range(num_pred)])

        self.reset_memory()


    def init_weights(self):
        if self.pretrained_path is not None:
            self.load_state_dict(torch.load(self.pretrained_path)['state_dict'],False)

    def reset_memory(self):
        self.memory_egopose = None
        self.sample_time = None
        self.memory_canbus = None

    def pre_update_memory(self, data):
        B = data['img_feats'].size(0)
        if self.memory_canbus is None:
            self.memory_egopose = data['img_feats'].new_zeros(B, self.can_bus_len, 4, 4)
            self.sample_time = data['timestamp'].new_zeros(B)
            if self.use_commandemb:
                self.memory_canbus = data['img_feats'].new_zeros(B, self.can_bus_len, 13)
            else:
                self.memory_canbus = data['img_feats'].new_zeros(B, self.can_bus_len, 14)
            x = self.sample_time.to(data['img_feats'].dtype)
        else:
            self.sample_time += data['timestamp']
            x = (torch.abs(self.sample_time) < 2.0).to(data['img_feats'].dtype)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.can_bus_len], x)
            self.memory_canbus = memory_refresh(self.memory_canbus[:, :self.can_bus_len], x)
            self.sample_time = data['timestamp'].new_zeros(B)
        
            
    def post_update_memory(self, data, rec_can_bus):
        B = data['img_feats'].size(0)
        rec_ego_pose = torch.eye(4, device=data['img_feats'].device).unsqueeze(0).unsqueeze(1).repeat(B,1,1,1)
        _rec_can_bus = rec_can_bus[..., :14]
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_canbus = torch.cat([_rec_can_bus, self.memory_canbus], dim=1)        
        self.sample_time -= data['timestamp']
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
        
            
    def mlp_position_encoding(self):
        pos_tensor = torch.arange(self.num_extra + 1, dtype=torch.float32).cuda().reshape(-1,1)
        pos_embed = self.mlp_position_encoder(pos_tensor)
        return pos_embed[:1], pos_embed[1:]
    
    def _mlp_position_encoding(self, num_channel):
        pos_tensor = torch.arange(num_channel, dtype=torch.float32).cuda().reshape(-1,1)
        pos_embed = self.mlp_position_encoder(pos_tensor)
        return pos_embed

    def get_query_embedding(self, batch_size, data):
        # TODO, 这个 pos embedding 384 
        # pos_embed_zero, pos_embed_4096 = self.mlp_position_encoding() # 作为 query position
        self.pre_update_memory(data)
        
        can_bus = data['can_bus']
        command = data['command'].unsqueeze(-1)
        can_bus_current = torch.cat([command, can_bus], dim=-1) # B, 14

        can_bus_memory = self.memory_canbus.flatten(-2)
        ego_pos_memory = self.memory_egopose.flatten(-3)
        can_bus_input = torch.cat([can_bus_current, can_bus_memory, ego_pos_memory], dim=-1) # B, 14 + can_bus_len*14 + can_bus_len*16 = 74
        
        can_bus_input = can_bus_input.unsqueeze(1)  # B, 1, 74
        can_bus_embedding = self.can_bus_embed(can_bus_input) # B, 1, C
        
        plan_anchor = torch.tile(self.plan_anchor[None], (batch_size, 1, 1, 1, 1))
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :], self.embed_dims)

        plan_pos = plan_pos.to(plan_anchor.dtype)
        plan_embed = self.plan_anchor_encoder(plan_pos)
        plan_embed = plan_embed.reshape(batch_size, -1, self.embed_dims)
        
        temp_embed = self.temporal_embedding.weight

        plan_embed = plan_embed.unsqueeze(2) + temp_embed.unsqueeze(0).unsqueeze(0)
        plan_embed = plan_embed.reshape(batch_size, -1, self.embed_dims)
        
        plan_pos_embed = self._mlp_position_encoding(plan_embed.shape[1])
        plan_pos_embed = plan_pos_embed.unsqueeze(0).repeat(batch_size,1,1)

        query_embedding = torch.cat([can_bus_embedding, plan_embed], dim=1)
        query_pos = torch.cat([can_bus_embedding*0.0, plan_pos_embed], dim=1)
        return query_embedding, query_pos, can_bus_input # query, pos, reference

    
    def forward(
        self, 
        plan_embed, # 目前的想法是一共18个query，对应 3个command * 6个时间步; B, 18, embed_dims(4096)
        data,
        img_metas=None,
        **kwargs,
    ):   
        all_layer_losses = {}
        if self.training:
            for layer_idx in range(len(self.plan_reg_branches)):
                layer_outs = self.forward_single_layer(
                    layer_idx,
                    plan_embed[layer_idx],
                    data,
                    img_metas=img_metas,
                    **kwargs,
                )
                all_layer_losses.update({f'layer{layer_idx}_'+k:v for k,v in layer_outs.items()})
        else:
            # only return last layer outputs during inference
            layer_outs = self.forward_single_layer(
                len(self.plan_reg_branches)-1,
                plan_embed,
                data,
                img_metas=img_metas,
                **kwargs,
            )
            plan_reg_pred = layer_outs
                
        self.post_update_memory(data, kwargs['reference_points']) # reference_points is can_bus
        
        if self.training:
            return all_layer_losses
        else:
            return plan_reg_pred
    
    def forward_single_layer(
        self, 
        layer_idx,
        plan_embed, # 目前的想法是一共18个query，对应 3个command * 6个时间步; B, 18, embed_dims(4096)
        data,
        img_metas=None,
        **kwargs,
    ):   
        B = plan_embed.shape[0]
        can_bus = kwargs['reference_points'] # hack, we use reference_points as interface to pass can_bus
        plan_embed_init = data['plan_embeds_before_llm']
        can_bus_embed_init = data['can_bus_embeds_before_llm']
        
        can_bus_embed = plan_embed[:, :1 , :]  # B, 1, C
        # plan_embed = plan_embed[:, 1: , :]  # B, num_plan_query
        plan_embed = self.input_norm[layer_idx](plan_embed[:, 1: , :])  # B, num_plan_query
        
        plan_embed = plan_embed + plan_embed_init + can_bus_embed + can_bus_embed_init
        
        plan_reg_pred = self.plan_reg_branches[layer_idx](plan_embed).reshape(B, self.ego_fut_mode, -1, 12)
        plan_cls = self.plan_cls_branches[layer_idx](plan_embed).squeeze(-1).reshape(B, self.ego_fut_mode, 6, -1)
        plan_status = self.plan_status_branches[layer_idx](can_bus_embed)
        
        # self.post_update_memory(data, can_bus)
        
        if self.training:
            losses = self.loss_planning(
                plan_cls, plan_reg_pred,plan_status, can_bus, data)
            return losses
        else:
            plan_cls = plan_cls.mean(dim=-1)  # B, ego_fut_mode, 6
            max_score_index = torch.argmax(plan_cls[torch.arange(B), data['command'].long()], dim=-1)      
            
            if self.using_gt_inference:
                gt_planning = torch.from_numpy(img_metas[0]['gt_planning'][...,:2]).to(plan_reg_pred.device)[None]
                tmp = plan_reg_pred[torch.arange(B), data['command'].long()].reshape(B, 6, 6, 2)
                res = gt_planning - tmp
                dist = torch.linalg.norm(res, dim=-1)
                dist = dist.mean(dim=-1)
                max_score_index_gt = torch.argmin(dist, dim=-1)
                max_score_index = max_score_index_gt
            
            plan_reg_pred = plan_reg_pred[torch.arange(B), data['command'].long(), max_score_index].reshape(B, 6, 2)
            return plan_reg_pred
    
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):
        loss = {}
        # if self.use_motion_loss:
        #     motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        #     loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        return loss

    # @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)
            
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    # @force_fp32(apply_to=("model_outs"))
    # def loss_planning(self, model_outs, data):
    def loss_planning(self, cls_scores, reg_preds, status_preds, can_bus, data):

        (
            cls,
            cls_target, 
            cls_weight, 
            reg_pred, 
            reg_target, 
            reg_weight, 
        ) = self.planning_sampler.sample(
            cls_scores,
            reg_preds,
            data['gt_planning'][...,:2],
            data['gt_planning_mask'],
            self.plan_anchor, 
            data,
        )
        cls = cls.flatten(end_dim=1)
        cls_target = cls_target.flatten(end_dim=1)
        cls_weight = cls_weight.flatten(end_dim=1)
        cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight.squeeze(1))
        reg_weight = reg_weight.flatten(end_dim=1)
        reg_pred = reg_pred.flatten(end_dim=1)
        reg_target = reg_target.flatten(end_dim=1)
        reg_weight = reg_weight.unsqueeze(-1)
        

        reg_loss = self.plan_loss_reg(
            reg_pred, reg_target.squeeze(1), weight=reg_weight[:,0,:,:].float()
        )
        reg_loss = torch.nan_to_num(reg_loss)
        cls_loss = torch.nan_to_num(cls_loss)
        
        # l1_reg_loss = F.l1_loss(reg_pred, reg_target.squeeze(1)).item()
        
        status_loss = self.plan_loss_status(status_preds, can_bus)
        
        # if status_loss.item() < 0.3:
        #     import ipdb; ipdb.set_trace()
        
        loss_weight = 1.0
        loss = {}
        loss.update(e2e_cls_loss=loss_weight*cls_loss, e2e_reg_loss=loss_weight*reg_loss, e2e_status_loss=loss_weight*status_loss)
        return loss

    def loss_planning_v2(self, cls_scores, reg_preds, status_preds, can_bus, data):
        gt_reg_target = data['gt_planning'][...,:2]
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = data['gt_planning_mask'].unsqueeze(1).any(dim=-1)

        if self.pred_res:
            # Calculate distances between target and anchors
            # import pdb;pdb.set_trace()
            bs_indices = torch.arange(gt_reg_mask.shape[0], device=gt_reg_mask.device)
            cmd_indices = data['command'].long()
            plan_anchor = self.plan_anchor[None].repeat(bs_indices.shape[0],1,1,1,1)[bs_indices, cmd_indices]
            dist = torch.linalg.norm(gt_reg_target.squeeze(1) - plan_anchor, dim=-1)
            dist = dist * gt_reg_mask[:,0,0,:,None]
            dist = dist.mean(dim=-2)  # Average over timesteps
            mode_idx = torch.argmin(dist, dim=-1)  # Best matching anchor
            # Get residuals between target and best anchor

            # Select the best anchor based on command and mode_idx
            best_anchor = plan_anchor[bs_indices, mode_idx]  # Shape should match gt_reg_target
            reg_target = (gt_reg_target - best_anchor[:,None,None])  # Residual target
            
            # Calculate losses
            cls = cls_scores.flatten(end_dim=1)
            cls_target = mode_idx.flatten()
            cls_weight = gt_reg_mask.any(dim=-1).flatten()
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)
            
            reg_pred = reg_preds[bs_indices, :, None, mode_idx.squeeze(-1)]
            reg_loss = self.plan_loss_reg(reg_pred.flatten(end_dim=1),reg_target.flatten(end_dim=1),weight=gt_reg_mask[:,0].unsqueeze(-1).float())
            
        else:
            # Original non-residual implementation
            cls_target = get_cls_target(reg_preds, gt_reg_target, gt_reg_mask)
            cls_weight = gt_reg_mask.any(dim=-1)
            reg_pred = get_best_reg(reg_preds, gt_reg_target, gt_reg_mask)
            cls = cls_scores.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight.squeeze(1))
            reg_weight = gt_reg_mask.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = gt_reg_target.flatten(end_dim=1)
            reg_weight = gt_reg_mask.unsqueeze(-1)
            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target.squeeze(1), weight=reg_weight[:,0,:,:].float()
            )

        status_loss = self.plan_loss_status(status_preds.squeeze(1), can_bus)
        loss_weight = 1.0
        loss = {}
        loss.update(e2e_cls_loss=loss_weight*cls_loss, 
                   e2e_reg_loss=loss_weight*reg_loss, 
                   e2e_status_loss=loss_weight*status_loss)
        return loss


    # @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )

        return motion_result, planning_result

class PlanningTarget():
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        use_anchor_match=False,
    ):
        super(PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.use_anchor_match = use_anchor_match

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        anchors,
        data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1).any(dim=-1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['command'].long()

        cls_pred = cls_pred.reshape(bs, 3, 1, self.ego_fut_mode, self.ego_fut_ts).mean(-1)
        reg_pred = reg_pred.reshape(bs, 3, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        cls_weight = gt_reg_mask.any(dim=-1)
        
        if self.use_anchor_match:
            anchor = anchors.repeat(bs,1,1,1,1).unsqueeze(2)
            anchor = anchor[bs_indices, cmd]
            cls_target = get_cls_target(anchor, gt_reg_target, gt_reg_mask)
            best_reg = reg_pred.squeeze(1)[bs_indices, cls_target.squeeze(-1)].unsqueeze(1)
        else:
            cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
            best_reg = get_best_reg(reg_pred, gt_reg_target, gt_reg_mask)

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def get_cls_target(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds
    reg_target_cum = reg_target
    dist = torch.linalg.norm(reg_target_cum - reg_preds_cum, dim=-1)
    dist = dist * reg_weight
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    return mode_idx

def get_best_reg(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds
    reg_target_cum = reg_target
    dist = torch.linalg.norm(reg_target_cum - reg_preds_cum, dim=-1)
    dist = dist * reg_weight
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
    return best_reg

def get_best_reg_with_idx(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds
    reg_target_cum = reg_target
    dist = torch.linalg.norm(reg_target_cum - reg_preds_cum, dim=-1)
    dist = dist * reg_weight
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
    return best_reg, mode_idx