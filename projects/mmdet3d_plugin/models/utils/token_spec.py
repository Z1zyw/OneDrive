from dataclasses import dataclass
from enum import IntEnum
import torch

# =========================
# Token Semantic Dimensions
# =========================

class Modality(IntEnum):
    TEXT = 0
    IMAGE = 1
    ABSTRACT = 2  # queries


class Role(IntEnum):
    CONTEXT = 0   # image / text / query_context(temporal etc.)
    QUERY = 1     # det / map / e2e

class BeginFlag(IntEnum):
    NON_BEGIN = 0
    BEGIN = 1

class Task(IntEnum):
    NONE = 0
    DET = 1
    MAP = 2
    E2E = 3
    TEXT = 4


# =========================
# TokenSpec Dataclass
# =========================

@dataclass
class TokenSpec:
    """
    A unified semantic specification for each token position.
    All tensors are shape [B, L].
    """
    modality: torch.LongTensor   # Modality
    role: torch.LongTensor       # Role
    task: torch.LongTensor       # Task
    begin_flag: torch.LongTensor  # Begin flag
    # temporal: torch.BoolTensor   # temporal attribute

    # ---------- basic masks ----------
    @ property
    def is_task(self, task: Task):
        return self.task == int(task)

    @ property
    def is_role(self, role: Role):
        return self.role == int(role)
    
    @ property
    def is_modality(self, modality: Modality):
        return self.modality == int(modality)

    # ---------- composite masks ----------
    @ property
    def is_det_query(self):
        return (self.task == Task.DET) & (self.role == Role.QUERY)
    
    @ property
    def is_det_token(self):
        return self.task == Task.DET

    @ property
    def is_map_query(self):
        return (self.task == Task.MAP) & (self.role == Role.QUERY)
    
    @ property
    def is_map_token(self):
        return self.task == Task.MAP
    
    @ property
    def is_e2e_query(self):
        return (self.task == Task.E2E) & (self.role == Role.QUERY)
    
    @ property
    def is_e2e_token(self):
        return self.task == Task.E2E
    
    @ property
    def is_temporal_token(self):
        return ((self.task == Task.DET) | (self.task == Task.MAP) | (self.task == Task.E2E)) & (self.role == Role.CONTEXT)
    
    @ property
    def is_begin_token(self):
        return self.begin_flag == BeginFlag.BEGIN
    
    @ staticmethod
    def mask2slice(mask):
        bzs, n = mask.shape
        pos = mask[0].nonzero(as_tuple=False)
        pos_beg, pos_end = pos.min(), pos.max() + 1
        
        assert pos_end - pos_beg == mask[0].sum()

        return slice(pos_beg, pos_end)


def init_from_metas(metas:dict, input_ids=None, query_mask=None):
    # query mask: instru(0), imgs(1), queries(1), prompt+text(0)
    assert input_ids is not None and query_mask is not None
    modality = torch.zeros_like(input_ids)
    role = torch.zeros_like(input_ids)
    task = torch.zeros_like(input_ids)
    begin_flag = torch.zeros_like(input_ids)
    
    query_begins = metas['query_begins'] 
    query_lens = metas['query_lens']      
    used_heads = metas['used_heads']   
    ref_lens = metas['ref_lens']      
    total_query_len = sum(query_lens)
    # img_begins_position = query_mask[0].nonzero(as_tuple=False)[:, 0].min()
    query_end_position = query_mask[0].nonzero(as_tuple=False)[:, 0].max() + 1
    query_beg_position = query_end_position - total_query_len
    
    modality[...] = Modality.TEXT
    modality[query_mask] = Modality.IMAGE
    modality[:, query_beg_position:query_end_position] = Modality.ABSTRACT
    
    # we define text, image, and temporal token(query) as CONTEXT role
    role[...] = Role.CONTEXT
    for i, head in enumerate(used_heads):
        begin = query_beg_position + query_begins[i]
        end = begin + ref_lens[i]
        role[:, begin:end] = Role.QUERY
    
    task[...] = Task.TEXT
    task[:, :query_beg_position] = Task.NONE
    head2task = {
        'pts_bbox_head': Task.DET,
        'map_head': Task.MAP,
        'e2e_head': Task.E2E,
    }
    for i, head in enumerate(used_heads):
        begin = query_beg_position + query_begins[i]
        end = begin + query_lens[i]
        task[:, begin:end] = head2task[head]
        begin_flag[:, begin] = BeginFlag.BEGIN

    return TokenSpec(
        modality=modality,
        role=role,
        task=task,
        begin_flag=begin_flag
    )
    
