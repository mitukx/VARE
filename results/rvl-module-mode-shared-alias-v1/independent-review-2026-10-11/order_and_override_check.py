import torch
from torch import nn


def graph(order):
    root = nn.Module()
    shared = nn.Dropout(p=0.5)
    branches = {name: nn.Sequential(shared) for name in ("left", "right")}
    for name in order:
        setattr(root, name, branches[name])
    return root, shared


def set_snapshot_modes(root, last_parent_training):
    modules = dict(root.named_modules())
    root.training = False
    order = list(root._modules)
    # Set mixed flags directly so setup itself does not recurse through the alias.
    getattr(root, order[0]).training = False
    getattr(root, order[1]).training = last_parent_training
    shared = root.left[0]
    shared.training = False
    return {name: bool(module.training) for name, module in modules.items()}


def restore_named_modules(root, modes, direct):
    modules = dict(root.named_modules())
    for name, module in modules.items():
        if direct:
            module.training = modes[name]
        else:
            module.train(modes[name])


def check_graph(order, last_parent_training):
    root, shared = graph(order)
    modes = set_snapshot_modes(root, last_parent_training)
    restore_named_modules(root, modes, direct=False)
    old_modes = {name: module.training for name, module in root.named_modules()}
    torch.manual_seed(91)
    old_output = shared(torch.ones(256))
    old_exact_modes = old_modes == modes
    old_exact_output = torch.equal(old_output, torch.ones(256))

    restore_named_modules(root, modes, direct=True)
    direct_modes = {name: module.training for name, module in root.named_modules()}
    torch.manual_seed(91)
    direct_output = shared(torch.ones(256))
    direct_exact_modes = direct_modes == modes
    direct_exact_output = torch.equal(direct_output, torch.ones(256))

    last_name = [name for name in order if name in dict(root.named_modules())][-1]
    return {
        "registration_order": list(order),
        "last_branch": last_name,
        "last_parent_training": last_parent_training,
        "saved_modes": modes,
        "recursive_restore_exact_modes": old_exact_modes,
        "recursive_restore_exact_eval_dropout_output": old_exact_output,
        "direct_restore_exact_modes": direct_exact_modes,
        "direct_restore_exact_eval_dropout_output": direct_exact_output,
    }


class HookModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.extra_mode = "initial"

    def train(self, mode=True):
        self.extra_mode = "train" if mode else "eval"
        return super().train(mode)


def custom_override_check():
    root = nn.Module()
    root.hook = HookModule()
    root.training = False
    root.hook.training = False
    saved = {name: bool(module.training) for name, module in root.named_modules()}
    root.train(True)
    before_restore = {"root": root.training, "hook": root.hook.training, "extra_mode": root.hook.extra_mode}
    restore_named_modules(root, saved, direct=True)
    after_direct = {"root": root.training, "hook": root.hook.training, "extra_mode": root.hook.extra_mode}
    return {"saved_flags": saved, "after_training_mutation": before_restore, "after_direct_flag_restore": after_direct}


if __name__ == "__main__":
    print("torch", torch.__version__)
    for order in (("left", "right"), ("right", "left")):
        print("alias_case", check_graph(order, last_parent_training=True))
        print("masking_case", check_graph(order, last_parent_training=False))
    print("custom_override", custom_override_check())
