"""Activation-weighted SVD for Part2; no gradient/parameter training.

Whitening and balanced factors follow SVD-LLM section 3.1:
https://arxiv.org/html/2403.07378v5#S3.SS1
The paper's subsequent parameter-update/fine-tuning stage is not used.
"""

from dataclasses import asdict, dataclass
import sys
import time

import torch
from torch import nn

from common import replace_module, tensor_bytes
from gptq import InputHessian, _capture_inputs, fp32_matmul
from quantization import MixedPrecisionLinear


@dataclass(frozen=True)
class WhiteningConfig:
    samples: int = 128
    seqlen: int = 2048
    damp: float = 1e-4
    seed: int = 0


@torch.no_grad()
def whitening_cholesky(gram, damp):
    """L L.T = mean(x x.T) + jitter*I; returns L and actual jitter.

    Damping regularizes weak/unobserved activation directions. Increase it only
    if Cholesky fails numerically; after six attempts let torch report failure.
    These tensors are temporary and never part of inference storage.
    """
    covariance = (gram + gram.T) * 0.5
    jitter = damp * covariance.diagonal().mean().clamp_min(torch.finfo(gram.dtype).eps)
    covariance.diagonal().add_(jitter)
    for _ in range(6):
        lower, info = torch.linalg.cholesky_ex(covariance)
        if info.item() == 0:
            break
        covariance.diagonal().add_(9 * jitter)
        jitter *= 10
    else:
        lower = torch.linalg.cholesky(covariance)
    return lower, float(jitter.item())


@torch.no_grad()
def whitened_svd_linear(linear, rank, lower):
    """W[O,I], lower[I,I] -> Sequential(B[r,I], A[O,r]), plus diagnostics.

    W L = U diag(S) Vh, A = U_r sqrt(S_r),
    B = sqrt(S_r) Vh_r inv(L). A triangular solve avoids an explicit inverse.
    Truncation minimizes the input-Gram-weighted squared reconstruction error.
    Retained factors use the source dtype/device and independent allocations.
    """
    dtype = torch.float64 if linear.weight.dtype == torch.float64 else torch.float32
    with fp32_matmul():
        weighted = linear.weight.detach().to(dtype) @ lower.to(dtype)
        u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
        del weighted
        root = singular[:rank].sqrt()
        a = u[:, :rank] * root
        right = root[:, None] * vh[:rank]
        b = torch.linalg.solve_triangular(lower.to(dtype).T, right.T, upper=True).T
        options = {"device": linear.weight.device, "dtype": linear.weight.dtype}
        first = nn.Linear(linear.in_features, rank, bias=False, **options)
        second = nn.Linear(rank, linear.out_features, bias=linear.bias is not None, **options)
        first.weight.copy_(b)
        second.weight.copy_(a)
        if linear.bias is not None:
            second.bias.copy_(linear.bias)
        energy = singular.square()
        discarded = energy[rank:].sum() / energy.sum().clamp_min(torch.finfo(dtype).tiny)
    result = nn.Sequential(first, second).requires_grad_(False).train(linear.training)
    return result, {"discarded_regularized_energy_fraction": float(discarded.item())}


@torch.no_grad()
def apply_whitened_svd(model, ranks, configs, row_plan, batches, config, *, provenance=None):
    """Sequential Qwen decoder calibration, whitening, then factor packing.

    Parallel q/k/v share a Gram, as do gate/up. Each subsequent group is
    calibrated using already-compressed predecessors. Only block inputs and
    the current group's Gram/Cholesky are retained, all on the model device.
    """
    sequence = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                ("self_attn.o_proj",), ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",))
    started = time.monotonic()
    model.eval()
    hidden, kwargs = _capture_inputs(model, batches, config)
    diagnostics = []
    for block_index, block in enumerate(model.model.layers):
        prefix = f"model.layers.{block_index}."
        for group in sequence:
            linear = block.get_submodule(group[0])
            accumulator = InputHessian(linear.in_features, hidden.device)
            handle = linear.register_forward_pre_hook(lambda module, args: accumulator.add(args[0]))
            try:
                for sample in range(config.samples):
                    block(hidden[sample:sample+1], **kwargs)
                    if (sample + 1) % 32 == 0:
                        print(f"[svd whitening block {block_index+1}/{len(model.model.layers)}] "
                              f"{group[0]} calibration {sample+1}/{config.samples}",
                              file=sys.stderr, flush=True)
            finally:
                handle.remove()
            tokens = accumulator.tokens
            gram = accumulator.gram.div_(tokens)
            with fp32_matmul():
                lower, jitter = whitening_cholesky(gram, config.damp)
            del gram, accumulator, linear
            for name in group:
                path = prefix + name
                quant_config, rank = configs[path], ranks[path]
                factors, stats = whitened_svd_linear(block.get_submodule(name), rank, lower)
                svd_bytes = sum(layer.weight.numel() * layer.weight.element_size() for layer in factors)
                rows = [row_plan[path, factor] for factor in (0, 1)]
                print(f"[whitened svd {len(diagnostics)+1}/{len(ranks)}] {path}: "
                      f"rank={rank}, quant={quant_config.bits}-bit, quantized rows={rows}",
                      file=sys.stderr, flush=True)
                for factor in (0, 1):
                    factors[factor] = MixedPrecisionLinear.from_linear(factors[factor], quant_config, rows[factor])
                diagnostics.append({"name": path, "rank": rank, "bits": quant_config.bits,
                                    "group_size": quant_config.group_size,
                                    "qparam_method": quant_config.qparam_method,
                                    "svd_factor_bytes": svd_bytes, "quantized_rows": rows,
                                    "final_bytes": tensor_bytes(factors), "calibration_tokens": tokens,
                                    "whitening_jitter": jitter, **stats})
                replace_module(model, path, factors)
                del factors
            del lower
        for sample in range(config.samples):
            output = block(hidden[sample:sample+1], **kwargs)
            if isinstance(output, tuple):
                output = output[0]
            hidden[sample:sample+1].copy_(output)
        del output
        print(f"[svd whitening block {block_index+1}/{len(model.model.layers)} complete] "
              f"elapsed={time.monotonic()-started:.1f}s", file=sys.stderr, flush=True)
    del hidden, kwargs
    metadata = {"algorithm": "SVD-LLM section 3.1 whitening with diagonal damping",
                "settings": asdict(config), "calibration": provenance or {},
                "sequential": True, "factor_scaling": "sqrt_singular_values",
                "seconds": time.monotonic()-started}
    return model, diagnostics, metadata
