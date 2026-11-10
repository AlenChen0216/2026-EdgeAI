"""Target selection and measured tensor bytes, not checkpoint size/GPU memory."""

import re

from torch import nn


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_TARGET = re.compile(
    r"model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$")


def target_linears(model):
    return {name: module for name, module in model.named_modules(remove_duplicate=False)
            if _TARGET.fullmatch(name) and isinstance(module, nn.Linear)}


def tensor_bytes(module):
    # Deduplicate identical tensor objects within a target. Distinct target
    # invocations are counted separately even when they share weights.
    tensors = list(module.parameters()) + list(module.buffers())
    return sum(t.numel() * t.element_size() for t in {id(t): t for t in tensors}.values())


def record_base(model):
    """Once on original model; retain numbers/names, never weight copies."""
    targets = target_linears(model)
    model._linear_base_bytes = {name: tensor_bytes(layer) for name, layer in targets.items()}
    # SVD uses W matrices only; overall CR continues to include bias/buffers.
    model._weight_base_bytes = {
        name: layer.weight.numel() * layer.weight.element_size()
        for name, layer in targets.items()
    }
    model._base_linear_numel = sum(layer.weight.numel() for layer in targets.values())
    model._base_tensor_names = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))


def storage_report(model):
    rows = []
    for name, original in model._linear_base_bytes.items():
        module = model.get_submodule(name)  # Missing targets are never silently skipped.
        rows.append({"name": name, "original_bytes": original,
                     "compressed_bytes": tensor_bytes(module), "type": type(module).__name__})
    extras = {}
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    for name, value in tensors.items():
        if name not in model._base_tensor_names and not any(
            name.startswith(path + ".") for path in model._linear_base_bytes
        ):
            extras[name] = value.numel() * value.element_size()
    original = sum(row["original_bytes"] for row in rows)
    compressed = sum(row["compressed_bytes"] for row in rows) + sum(extras.values())
    svd = getattr(model, "_svd_measurement", None)
    pruning = getattr(model, "_pruning_measurement", None)
    if svd is not None:
        retained = svd["factor_bytes"] / svd["original_bytes"]
        svd = dict(svd, compression_ratio=1 - retained, retained_ratio=retained,
                   definition=getattr(model, "_svd_stage_definition",
                                      "1 - SVD-only weight representation / original W; all targets"))
    if pruning is not None:
        reduction = 1 - pruning["compressed_bytes"] / pruning["original_bytes"]
        pruning = dict(pruning, rate=reduction, compression_ratio=reduction,
                       element_fraction=pruning["pruned_elements"] / pruning["original_elements"])
    return {"layers": rows, "extra_bytes": extras, "original_bytes": original,
            "compressed_bytes": compressed, "compression_ratio": 1 - compressed / original,
            "svd": svd, "pruning": pruning}


def compression_rate(model):
    return 0.0 if not hasattr(model, "_linear_base_bytes") else storage_report(model)["compression_ratio"]


def replace_module(model, name, replacement):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent), leaf, replacement)
