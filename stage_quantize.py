"""Stage-aware (heterogeneous) quantization for CSM.

Applies *different* torchao quantization recipes to the different stages of
a CSM ``Model`` -- the autoregressive Llama ``backbone`` and the smaller
audio ``decoder`` -- so the compute-heavy language model gets the most
aggressive low-bit treatment while the acoustic decoder keeps a milder
precision. This is the core insight of *VibeVoice-ASR-BitNet*
(arXiv:2607.21075), which compresses a VAE-acoustic-tokenizer +
autoregressive-LM speech model for real-time edge-CPU inference by giving
each stage a quantization recipe matched to its compute profile (full INT8
for the acoustic stage, ternary weights for the LM).

This is an **adapted port (Mode 2)**. CSM is a frozen-weights PyTorch
inference repo, so the paper's components that need training or a
non-PyTorch runtime are substituted with target-native equivalents:

* BitNet-style ternary LM weights (I2_S)  -> INT4 weight-only quantization.
  Ternary weights need quantization-aware training, which is unavailable in
  CSM's frozen-weights regime; INT4 weight-only is the closest
  retraining-free low-bit option for the LM stage.
* full-pipeline INT8 + custom ggml SIMD kernels for the acoustic stage
  -> torchao INT8 weight-only quantization. CSM has no ggml runtime, so the
  acoustic decoder (a torchtune transformer) is quantized with the repo's
  already-pinned ``torchao`` (==0.9.0) instead.
* progressive quantization-aware training -> cut entirely (frozen weights).

The progressive-QAT accuracy recovery and the ggml SIMD speedups are
intentionally out of scope; what remains is the paper's stage-aware *recipe
assignment* delivered as a retraining-free capability on CSM's real
``Model``, plus a CPU timing helper for the suggested RTF-vs-baseline
experiment.

Example
-------
>>> from generator import load_csm_1b
>>> from stage_quantize import quantize_model
>>> generator = load_csm_1b(device="cpu")
>>> report = quantize_model(generator._model)
>>> report["weight_compression_ratio_est"]  # weight bytes shrank ~3-4x
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch.nn import Linear

# recipe name -> (torchao config class name, bytes per stored weight element).
# The byte counts drive an honest compression estimate computed from the
# model's own Linear parameter counts, so the report does not depend on
# torchao's internal packed-storage layout.
_RECIPES: dict[str, tuple[str | None, float | None]] = {
    "int4_weight_only": ("Int4WeightOnlyConfig", 0.5),
    "int8_weight_only": ("Int8WeightOnlyConfig", 1.0),
    "none": (None, None),
}

# Default stage -> recipe map. The autoregressive LM (backbone) is the
# dominant compute and memory consumer, so it gets the most aggressive
# low-bit recipe -- mirroring the paper's choice of the lowest bit-width
# (ternary) for the LM. The acoustic decoder keeps milder INT8, mirroring
# the paper's full INT8 for the VAE/acoustic stage.
DEFAULT_STAGE_RECIPES: dict[str, str] = {
    "backbone": "int4_weight_only",
    "decoder": "int8_weight_only",
}


def _resolve_recipe(recipe: str) -> tuple[Callable | None, float | None]:
    """Return ``(config_factory_or_None, bytes_per_element_or_None)``.

    ``config_factory`` is a zero-arg callable returning a fresh torchao
    config; torchao is imported lazily so this module only needs ``torch``
    at import time (matching ``models.py``).
    """
    if recipe not in _RECIPES:
        raise ValueError(f"Unknown recipe {recipe!r}; choose from {sorted(_RECIPES)}")
    config_attr, bytes_per_element = _RECIPES[recipe]
    if config_attr is None:
        return None, None
    from torchao import quantization as _ao_quantization

    config_cls = getattr(_ao_quantization, config_attr)
    return config_cls, bytes_per_element


def _linears(module: torch.nn.Module) -> list[Linear]:
    found: list[Linear] = []
    module.apply(lambda sub: found.append(sub) if isinstance(sub, Linear) else None)
    return found


def _is_quantized(linear: Linear) -> bool:
    """True if torchao has swapped this Linear's weight for a quantized tensor.

    Detection is by type name so it survives minor torchao refactors rather
    than importing a private symbol.
    """
    weight_type = type(linear.weight).__name__
    return weight_type == "AffineQuantizedTensor" or "QuantizedTensor" in weight_type


def _quantize_stage(module: torch.nn.Module, recipe: str) -> dict:
    """Apply *recipe* to every Linear in *module* (in place) and report.

    The report records how many Linears were actually quantized. If the
    current backend cannot host a recipe, ``linears_quantized`` will be less
    than ``linears_total`` -- the caller sees the gap rather than a silent
    no-op.
    """
    linears = _linears(module)
    baseline_bpe = int(linears[0].weight.element_size()) if linears else 2
    total_numel = sum(int(linear.weight.numel()) for linear in linears)
    before_bytes = total_numel * baseline_bpe

    config_factory, recipe_bpe = _resolve_recipe(recipe)
    applied_bpe = recipe_bpe if recipe_bpe is not None else baseline_bpe
    after_bytes = total_numel * applied_bpe

    if config_factory is not None:
        from torchao.quantization import quantize_

        quantize_(module, config_factory())

    quantized = sum(1 for linear in _linears(module) if _is_quantized(linear))
    return {
        "recipe": recipe,
        "linears_quantized": quantized,
        "linears_total": len(linears),
        "weight_bytes_before": before_bytes,
        "weight_bytes_after_est": after_bytes,
        "weight_compression_ratio_est": (
            before_bytes / after_bytes if after_bytes else float("inf")
        ),
    }


def quantize_model(
    model: torch.nn.Module,
    stage_recipes: dict[str, str] | None = None,
) -> dict:
    """Apply stage-aware quantization to a CSM ``Model`` (in place).

    Each stage (``model.backbone``, ``model.decoder``) is mapped to a torchao
    recipe via *stage_recipes* (defaults to :data:`DEFAULT_STAGE_RECIPES`).
    Returns a report describing, per stage, the recipe applied, how many
    Linears were quantized, and an estimated weight-byte footprint before
    and after.
    """
    if stage_recipes is None:
        stage_recipes = DEFAULT_STAGE_RECIPES

    stages: dict[str, dict] = {}
    total_before = total_after = 0
    for stage_name in ("backbone", "decoder"):
        module = getattr(model, stage_name, None)
        if module is None:
            continue
        report = _quantize_stage(module, stage_recipes.get(stage_name, "none"))
        stages[stage_name] = report
        total_before += report["weight_bytes_before"]
        total_after += report["weight_bytes_after_est"]

    return {
        "stages": stages,
        "weight_bytes_before": total_before,
        "weight_bytes_after_est": total_after,
        "weight_compression_ratio_est": (
            total_before / total_after if total_after else float("inf")
        ),
    }


def time_forward(
    fn: Callable,
    *args,
    repeats: int = 5,
    warmup: int = 2,
    **kwargs,
) -> float:
    """Median wall-clock seconds to call ``fn(*args, **kwargs)`` on CPU.

    Runs under ``torch.no_grad()``. Use it to measure the inference speedup
    of a quantized stage against its baseline -- the CPU RTF-vs-baseline
    comparison suggested for this capability. RTF for a given stage is
    ``time_forward(...) / real_time_of_the_input``.
    """
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args, **kwargs)
        samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn(*args, **kwargs)
            samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]
