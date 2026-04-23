from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FreezeHook(Hook):
    def __init__(self, target_modules=('backbone',)):
        self.target_modules = target_modules

    def before_run(self, runner):
        for name, module in runner.model.named_modules():
            if any(k in name for k in self.target_modules):
                for p in module.parameters():
                    p.requires_grad = False
                print(f"[FreezeHook] Freezing module: {name}") # for debug
        print(f"[FreezeHook] Frozen modules: {self.target_modules}")
        # for param_group in runner.optim_wrapper.optimizer.param_groups:
        #     if any(k in param_group.get('group_name', '') for k in self.target_modules):
        #         old_lr = param_group['lr']
        #         base_lr = param_group.get('lr', runner.optim_wrapper.optimizer.defaults['lr'])
        #         param_group['lr'] = self.freeze_lr_scale * base_lr
        #         # 确保 requires_grad=True 避免 DDP 报错
        #         for p in param_group['params']:
        #             p.requires_grad = True
        #         print(f"[DDPFreezeHook] Freeze group {param_group['group_name']} lr {old_lr:.2e} -> {param_group['lr']:.2e}")

@HOOKS.register_module()
class UnfreezeHook(Hook):
    def __init__(self, start_iter=2000, target_modules=('backbone',)):
        self.start_iter = start_iter
        self.target_modules = target_modules
        self.updated = False

    def after_train_iter(self, runner):
        if not self.updated and runner.iter >= self.start_iter:
            for name, module in runner.model.named_modules():
                if any(k in name for k in self.target_modules):
                    for p in module.parameters():
                        p.requires_grad = True
                print(f"[UnfreezeHook] Unfreezing module: {name}") # for debug
            print(f"[UnfreezeHook] Unfrozen modules at iter {runner.iter}: {self.target_modules}")
            self.updated = True