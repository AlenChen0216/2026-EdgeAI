"""Part 2 GPTQ using torch.linalg and the existing packed inference format.

Algorithm: Frantar et al., GPTQ (ICLR 2023), Algorithm 1.
https://arxiv.org/abs/2210.17323
Reference behavior: https://github.com/IST-DASLab/gptq

This implementation handles nn.Linear only. It uses original column order,
dynamic contiguous groups, damped input Gram matrices and sequential decoder
calibration. No activation-order permutation, custom CUDA extension, gradient
training, parameter offloading or persistent dense weight copy is required.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import glob
import hashlib
import math
import os
import random
import time

import torch
from torch import nn

from common import replace_module, target_linears, tensor_bytes
from quantization import (CompactPackedLinear, PackedLinear, Sparse34PackedLinear, QuantConfig, choose_qparams,
                          choose_qparams_mse, pack_codes)


@dataclass(frozen=True)
class GPTQConfig:
    samples: int = 128
    seqlen: int = 2048
    damp: float = 0.01
    block_size: int = 128
    seed: int = 0


@contextmanager
def fp32_matmul():
    """Match GPTQ's FP32 Hessian arithmetic, restoring caller settings on error."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class InputHessian:
    """Streaming H=2/N sum(x x^T), where N counts calibration tokens."""

    def __init__(self, features, device):
        self.gram = torch.zeros(features, features, device=device, dtype=torch.float32)
        self.tokens = 0

    @torch.no_grad()
    def add(self, inputs):
        x = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        with fp32_matmul():
            self.gram.addmm_(x.T, x)
        self.tokens += x.shape[0]

    def hessian(self):
        return self.gram * (2.0 / self.tokens)


@torch.no_grad()
def gptq_linear(linear, hessian, config, *, damp=0.01, block_size=128,
                pruning=0.0, compact=False, pruning_pattern=None):
    """Return (PackedLinear, diagnostics), consuming no persistent Hessian state.

    W: [O,I], H: [I,I]. The upper Cholesky factor R of H^-1 gives
    err=(w_j-q_j)/R[j,j], then W_remaining -= err * R[j,remaining].
    Rank-one updates inside each block are followed by a matrix tail update.
    Blocks stop at group boundaries so dynamic qparams use corrected weights.
    Padding never enters H or the qparam search and is encoded as exact zero.
    With pruning, select low w^2/R_jj^2 scores within each column group and
    jointly compensate pruning/quantization error, as in SparseGPT.
    """
    compact = compact or config.compact
    rows, cols = linear.weight.shape
    if pruning_pattern is not None and pruning_pattern != "3:4":
        raise ValueError("unsupported pruning pattern")
    sparse34 = pruning_pattern == "3:4"
    if sparse34 and (pruning != 0.25 or not compact or cols % 4 or config.group_size % 4):
        raise ValueError("3:4 pruning requires fraction=0.25, compact storage and widths/groups divisible by 4")
    with fp32_matmul():
        work = linear.weight.detach().float().clone()
        h = hessian.float().clone()
        dead = h.diagonal() == 0
        # Unobserved columns contribute no calibration loss. Give them a finite
        # diagonal and zero their weights, as in the reference GPTQ algorithm.
        h.diagonal()[dead] = 1
        work[:, dead] = 0
        damping = damp * h.diagonal().mean()
        h.diagonal().add_(damping)
        try:
            inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
            factor = torch.linalg.cholesky(inverse, upper=True)
        except torch.linalg.LinAlgError as error:
            raise ValueError("GPTQ Hessian factorization failed; increase --gptq-damp. "
                             "No quantization fallback was applied.") from error
        del h, inverse

        groups = math.ceil(cols / config.group_size)
        codes = torch.empty(rows, groups * config.group_size, device=work.device,
                            dtype=torch.uint8)
        scales = torch.empty(rows, groups, 1, device=work.device, dtype=torch.float32)
        zeros = torch.empty_like(scales)
        loss = torch.zeros((), device=work.device)
        pruned_elements = 0
        omitted = torch.empty(rows, cols // 4, device=work.device,
                              dtype=torch.uint8) if sparse34 else None
        for group in range(groups):
            begin = group * config.group_size
            end = min(begin + config.group_size, cols)
            if pruning:
                original_nonzero = linear.weight[:, begin:end] != 0
                scores = (work[:, begin:end].square()
                          / factor.diagonal()[begin:end].square())
                scores.masked_fill_(~original_nonzero, torch.inf)
                mask = torch.zeros_like(scores, dtype=torch.bool)
                if sparse34:
                    positions = scores.reshape(rows, -1, 4).argmin(-1, keepdim=True)
                    mask.reshape(rows, -1, 4).scatter_(-1, positions, True)
                    omitted[:, begin//4:end//4] = positions.squeeze(-1).to(torch.uint8)
                    count = int(original_nonzero[mask].sum().item())
                else:
                    count = min(math.ceil(scores.numel() * pruning),
                                int(original_nonzero.sum().item()))
                    mask.view(-1)[scores.flatten().topk(count, largest=False).indices] = True
                pruned_elements += count
            values = work[:, begin:end].unsqueeze(1)
            valid = torch.ones_like(values, dtype=torch.bool)
            if config.qparam_method == "mse":
                scale, zero = choose_qparams_mse(
                    values, valid, config.bits, grid_size=config.mse_grid_size,
                    min_ratio=config.mse_min_ratio)
            else:
                scale, zero = choose_qparams(values, valid, config.bits)
            if compact:
                # Compensate using the same rounded scale that inference stores.
                scale = scale.half().clamp_min(torch.finfo(torch.float16).tiny).float()
            scales[:, group:group+1] = scale
            zeros[:, group:group+1] = zero
            scale, zero = scale[:, 0, 0], zero[:, 0, 0]
            for start in range(begin, end, block_size):
                stop = min(start + block_size, end)
                local = work[:, start:stop].clone()
                errors = torch.empty_like(local)
                for j in range(stop - start):
                    column = local[:, j]
                    code = ((column / scale).round() + zero).clamp(0, 2**config.bits - 1)
                    if pruning:
                        code = torch.where(mask[:, start+j-begin], zero, code)
                    codes[:, start+j] = code.to(torch.uint8)
                    quantized = (code - zero) * scale
                    error = (column - quantized) / factor[start+j, start+j]
                    errors[:, j] = error
                    local[:, j:].addcmul_(error[:, None],
                                         factor[start+j, start+j:stop][None, :], value=-1)
                work[:, stop:].addmm_(errors, factor[start:stop, stop:], beta=1, alpha=-1)
                loss += errors.square().sum() * 0.5
            if end < begin + config.group_size:
                codes[:, end:begin+config.group_size] = zero[:, None].to(torch.uint8)
        if not torch.isfinite(scales).all() or not torch.isfinite(loss):
            raise ValueError("GPTQ produced nonfinite parameters/loss")
        if sparse34:
            result = Sparse34PackedLinear(codes[:, :cols], scales, zeros,
                                          (rows, cols), config, omitted, linear.bias).eval()
        else:
            packed_type = CompactPackedLinear if compact else PackedLinear
            result = packed_type(pack_codes(codes, config.bits), scales, zeros,
                                 (rows, cols), config, linear.bias).eval()
        stats = {"dead_columns": int(dead.sum().item()),
                 "damping": float(damping.item()),
                 "damped_quadratic_loss": float(loss.item())}
        if pruning:
            stats.update(original_elements=rows * cols, pruned_elements=pruned_elements)
        if sparse34:
            # Same sparse representation BEFORE quantization: original-dtype
            # surviving values + actual packed positions + metadata + bias.
            # Count every byte; no qparams are needed in this pruning-only stage.
            pruning_bytes = (rows * cols // 4 * 3 * linear.weight.element_size()
                             + result.omitted.numel() * result.omitted.element_size()
                             + result.shape_bits_group.numel() * result.shape_bits_group.element_size())
            if result.bias is not None:
                pruning_bytes += result.bias.numel() * result.bias.element_size()
            stats.update(pruning_pattern="3:4", pruning_original_bytes=tensor_bytes(linear),
                         pruning_only_bytes=pruning_bytes)
        if compact:
            stats.update(scale_dtype="float16", zero_point_storage="bit_packed")
        return result, stats


def calibration_data(tokenizer, config):
    """Deterministic WikiText-2 TRAIN windows; never reads the test split."""
    from datasets import Dataset, load_dataset

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except (PermissionError, OSError, ConnectionError, ValueError) as error:
        cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
        candidates = sorted(glob.glob(os.path.join(
            cache, "wikitext/wikitext-2-raw-v1/*/*/wikitext-train.arrow")))
        if not candidates:
            raise RuntimeError("GPTQ needs WikiText-2 train data; no cached train split found") from error
        dataset = Dataset.from_file(candidates[-1])
    ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", verbose=False).input_ids
    rng = random.Random(config.seed)
    starts = [rng.randrange(ids.shape[1] - config.seqlen + 1) for _ in range(config.samples)]
    batches = [ids[:, start:start+config.seqlen].clone() for start in starts]
    digest = hashlib.sha256(torch.cat(batches).numpy().tobytes()).hexdigest()
    return batches, {"dataset": "wikitext", "dataset_config": "wikitext-2-raw-v1",
                     "split": "train", "dataset_fingerprint": dataset._fingerprint,
                     "window_starts": starts, "token_ids_sha256": digest}


class _CapturedInput(Exception):
    """Private sentinel: stop before the first decoder, avoiding LM-head logits."""


@torch.no_grad()
def _capture_inputs(model, batches, config):
    first = model.model.layers[0]
    device = next(model.parameters()).device
    hidden, saved_kwargs, index = None, None, 0

    def capture(module, args, kwargs):
        nonlocal hidden, saved_kwargs, index
        value = args[0] if args else kwargs["hidden_states"]
        if hidden is None:
            hidden = value.new_empty(config.samples, config.seqlen, value.shape[-1])
            saved_kwargs = {key: val for key, val in kwargs.items() if key != "hidden_states"}
        hidden[index].copy_(value[0])
        index += 1
        raise _CapturedInput

    handle = first.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for batch in batches:
            try:
                model.model(batch.to(device), use_cache=False)
            except _CapturedInput:
                continue
            raise RuntimeError("GPTQ failed to intercept the first decoder")
    finally:
        handle.remove()
    return hidden, saved_kwargs


@torch.no_grad()
def apply_gptq(model, configs, batches, config, *, provenance=None,
               linear_compressor=None):
    """Sequential Qwen3 blocks; all parameters/activation buffers stay on device.

    q/k/v share an input Hessian, as do gate/up. Replaying each block after a
    group replacement calibrates later projections against quantized predecessors.
    Only a single block-input activation buffer and current group H are retained.
    """
    if linear_compressor is None:
        linear_compressor = gptq_linear
    targets = target_linears(model)
    del targets
    sequence = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                ("self_attn.o_proj",), ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",))
    started = time.monotonic()
    model.eval()
    hidden, kwargs = _capture_inputs(model, batches, config)
    diagnostics = []
    for block_index, block in enumerate(model.model.layers):
        prefix = f"model.layers.{block_index}."
        for names in sequence:
            names = [name for name in names if configs[prefix+name] is not None]
            if not names:
                continue
            layer = block.get_submodule(names[0])
            accumulator = InputHessian(layer.in_features, hidden.device)
            # Members of these two parallel projection groups have identical inputs.
            handle = layer.register_forward_pre_hook(lambda module, args: accumulator.add(args[0]))
            try:
                for sample in range(config.samples):
                    block(hidden[sample:sample+1], **kwargs)
            finally:
                handle.remove()
            hessian = accumulator.hessian()
            tokens = accumulator.tokens
            del accumulator, layer
            for name in names:
                path, quant_config = prefix+name, configs[prefix+name]
                replacement, stats = linear_compressor(
                    block.get_submodule(name), hessian, quant_config,
                    damp=config.damp, block_size=config.block_size)
                replace_module(model, path, replacement)
                diagnostics.append({"name": path, "bits": quant_config.bits,
                                    "group_size": quant_config.group_size,
                                    "qparam_method": quant_config.qparam_method,
                                    "calibration_tokens": tokens, **stats})
            del hessian, replacement
        for sample in range(config.samples):
            output = block(hidden[sample:sample+1], **kwargs)
            if isinstance(output, tuple):
                output = output[0]
            hidden[sample:sample+1].copy_(output)
        del output
    del hidden, kwargs
    model._gptq_metadata = {"algorithm": "GPTQ", "settings": asdict(config),
                            "act_order": False, "groups": "dynamic_contiguous",
                            "sequential": True, "calibration": provenance or {},
                            "layers": diagnostics, "seconds": time.monotonic()-started}
    return model
