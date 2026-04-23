import torch
import torch.nn as nn

from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import LoraLayer

# TODO register model in other file
# from peft.peft_model import PEFT_MODELS_TO_LORA_CONFIG_MAPPING, PEFT_TYPE_TO_MODEL_MAPPING
# PEFT_MODELS_TO_LORA_CONFIG_MAPPING['MultiBranchLoraModel'] = 
    

class FFNMultiLoraLayer(LoraLayer):
    def __init__(self, base_layer, ephemeral_gpu_offload = False, **kwargs):
        super().__init__(base_layer, ephemeral_gpu_offload, **kwargs)
        
class MultiBranchLoraModel(LoraModel):
    def __init__(self, model: nn.Module, lora_config_list: list, **kwargs):
        self.lora_config_list = lora_config_list
        super().__init__(model, lora_config_list[0], **kwargs)
        self._register_multi_lora_layers()
        
    def _register_multi_lora_layers(self):
        for name, module in self.named_modules():
            for lora_config in self.lora_config_list:
                if isinstance(module, lora_config.base_model_class) and any(
                    target_module in name for target_module in lora_config.target_modules
                ):
                    # Replace the module with a LoraLayer
                    parent_module = self.get_submodule(".".join(name.split(".")[:-1]))
                    setattr(
                        parent_module,
                        name.split(".")[-1],
                        FFNMultiLoraLayer(module, **lora_config.to_dict()),
                    )
                    break  # No need to check other configs once matched

    def merge_and_unload(self):
        for name, module in self.named_modules():
            if isinstance(module, FFNMultiLoraLayer):
                # Merge LoRA weights into the base layer
                module.merge_weights()
                # Replace the LoraLayer with the original base layer
                parent_module = self.get_submodule(".".join(name.split(".")[:-1]))
                setattr(parent_module, name.split(".")[-1], module.base_layer)
        # Clear the list of LoRA layers
        self.lora_layers = nn.ModuleList()