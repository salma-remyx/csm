"""Fisher-weighted channel-wise sensitivity scaling for post-training quantization.

Adapted from C-PTQ (arXiv:2607.21076v1), "C-PTQ: Fisher-weighted Channel-wise
Sensitivity for Post-training Quantization of MLLMs".

C-PTQ's core idea: when post-training-quantizing an LLM/MLLM decoder, outlier
channels are highly sensitive to quantization. Channel-wise scaling (CWS)
redistributes the quantization budget across channels to protect them, but the
quality of the result depends on *how the per-channel scale is chosen*. Existing
methods pick the scale from modality- or token-level heuristics that are
orthogonal to the task loss. C-PTQ instead picks the scale by minimizing a
**Fisher-weighted objective**, where the Fisher information diagonal is used as a
tractable Hessian approximation of the task loss. This injects task sensitivity
into the scaling process and "harmonizes task-specific loss perturbation and
quantization error."

This module implements that core mechanism and folds the resulting per-channel
scales into a model as an exact float equivalence (weight scaled up, preceding
activation scaled down), so a downstream quantizer (the repo's existing
``torchao`` dependency) sees better-conditioned weights on task-sensitive
channels. It is **training-free** and drops into ``load_csm_1b`` without touching
``generate`` / ``generate_frame``.

Implementation mode: **Mode 2 (adapted port).** The Fisher-weighted channel
sensitivity objective and the channel-wise scaling application are kept at full
fidelity. The paper's auxiliary components are substituted as follows:
  * The paper's task-specific (image-caption / understanding) calibration loss
    and its per-layer gradient pipeline are replaced by a parameter-free
    activation-energy proxy by default; a real Fisher diagonal can be supplied
    when a task loss / gradient hook is available (see ``fisher_diagonal``).
  * The paper's separate MLLM benchmark suite (Qwen2.5VL / InternVL2 /
    LLaVA-OV) is cut -- evaluation is a downstream PR and does not apply to a
    speech model.
  * The final integer quantization itself is handed off to ``torchao``; this
    module produces and folds the per-channel *scales*, which is C-PTQ's
    contribution. ``torchao`` consumes ``module._fisher_act_scales``.
  * The expensive per-layer iterative reconstruction is replaced by a grid
    search over the standard CWS smoothness coefficient, selecting the
    candidate that minimizes the Fisher-weighted objective.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

__all__ = [
    "quantize_per_tensor_symmetric",
    "fisher_diagonal",
    "fisher_proxy",
    "fisher_weighted_error",
    "compute_layer_scales",
    "apply_layer_scaling",
    "fisher_scale_model",
]


def quantize_per_tensor_symmetric(weight: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Fake-quantize ``weight`` with a symmetric per-tensor uniform quantizer.

    The CWS regime that benefits from channel-wise scaling is per-tensor (or
    per-group) quantization, where scaling redistributes a *shared* quantization
    budget across channels; this simulator models that regime and is used only
    inside the scale-selection objective -- it never replaces the real
    ``torchao`` quantizer.
    """
    qmax = 2 ** (bits - 1) - 1
    scale = weight.abs().max() / qmax
    scale = scale.clamp(min=1e-8)
    return (weight / scale).round().clamp(-qmax, qmax) * scale


def fisher_diagonal(grads: torch.Tensor) -> torch.Tensor:
    """Diagonal Fisher information E[(dL/da)^2] over calibration samples.

    ``grads`` is (num_samples, in_features): the per-sample gradient of a task
    loss with respect to a layer's input activations. This is the tractable
    Hessian approximation C-PTQ uses to inject task sensitivity into scaling.
    Supply it when a task loss / gradient hook is available; otherwise fall back
    to the parameter-free :func:`fisher_proxy`.
    """
    return (grads.float() ** 2).mean(dim=0)


def fisher_proxy(activations: torch.Tensor) -> torch.Tensor:
    """Parameter-free Fisher surrogate: per-channel activation energy.

    Substitutes for the learned/task-specific Fisher estimator when no
    calibration gradient is available (Mode 2). This reduces to the AWQ-style
    activation-magnitude importance; supplying a real :func:`fisher_diagonal`
    is exactly the C-PTQ improvement on top of it.
    """
    return (activations.float() ** 2).mean(dim=0)


def fisher_weighted_error(
    weight: torch.Tensor,
    activations: torch.Tensor,
    scales: torch.Tensor,
    fisher: torch.Tensor,
    bits: int = 4,
) -> torch.Tensor:
    """C-PTQ objective: Fisher-weighted per-channel output reconstruction error.

    With per-input-channel scale ``s`` we form ``W~ = W * s`` and
    ``A~ = A / s`` so the layer output is unchanged in float; quantizing ``W~``
    perturbs the output by ``delta[n,o] = sum_c A[n,c] * eps[o,c] / s_c`` where
    ``eps = Q(W~) - W~``. Under the diagonal-Hessian approximation the output
    error energy decomposes per channel, and C-PTQ weights each channel by its
    Fisher diagonal ``F`` to measure its *task-loss* impact:

        obj(s) = sum_c F_c * (sum_n A[n,c]^2) * (sum_o (eps[o,c] / s_c)^2)

    Args:
        weight: (out_features, in_features) layer weight.
        activations: (num_samples, in_features) calibration inputs.
        scales: (in_features,) candidate per-channel scale.
        fisher: (in_features,) per-channel Fisher diagonal (or proxy).
        bits: target quantization width used by the simulated quantizer.

    Returns:
        Scalar Fisher-weighted error for ``scales``.
    """
    scales = scales.clamp(min=1e-4, max=1e4)
    scaled_weight = weight * scales  # (out, in)
    quantized = quantize_per_tensor_symmetric(scaled_weight, bits)
    eps_over_s = (quantized - scaled_weight) / scales  # effective per-channel error
    weight_term = (eps_over_s**2).sum(dim=0)  # sum over out_features -> (in,)
    act_term = (activations**2).sum(dim=0)  # sum over samples -> (in,)
    return (fisher * weight_term * act_term).sum()


def compute_layer_scales(
    weight: torch.Tensor,
    activations: torch.Tensor,
    fisher: Optional[torch.Tensor] = None,
    bits: int = 4,
    grid_size: int = 21,
) -> torch.Tensor:
    """Select per-channel scales by minimizing the Fisher-weighted objective.

    Candidates come from the standard CWS closed form
    ``s_c = a_max_c**alpha / w_max_c**(1-alpha)`` trading activation protection
    against weight protection; the selected ``alpha`` is the one that minimizes
    :func:`fisher_weighted_error`, which is where C-PTQ's task sensitivity enters
    (no ``fisher`` supplied -> activation-energy proxy, i.e. the AWQ baseline).
    """
    in_features = weight.shape[1]
    device, dtype = weight.device, weight.dtype
    weight = weight.double()
    activations = activations.double()
    if fisher is None:
        fisher = fisher_proxy(activations)
    fisher = fisher.double().to(device)

    identity = torch.ones(in_features, dtype=torch.double, device=device)
    best_scales = identity
    best_error = fisher_weighted_error(weight, activations, identity, fisher, bits)

    w_max = weight.abs().amax(dim=0).clamp(min=1e-8)
    a_max = activations.abs().amax(dim=0).clamp(min=1e-8)
    for alpha in torch.linspace(0.0, 1.0, grid_size, device=device):
        alpha = alpha.item()
        scales = (a_max**alpha) / (w_max ** (1.0 - alpha))
        scales = scales.clamp(min=1e-4, max=1e4)
        error = fisher_weighted_error(weight, activations, scales, fisher, bits)
        if error < best_error:
            best_error = error
            best_scales = scales

    return best_scales.float().to(device=device, dtype=dtype)


def apply_layer_scaling(layer: nn.Linear, scales: torch.Tensor) -> torch.Tensor:
    """Fold ``scales`` into ``layer`` as an exact float equivalence.

    Scales the weight up along the input-feature axis (``W * s``) and registers
    a ``forward_pre_hook`` that scales the incoming activation down (``A / s``),
    so the layer output is unchanged in float but a downstream quantizer sees
    the re-scaled weight distribution. The scale is also stored on
    ``layer._fisher_act_scales`` for a torchao quantizer to consume.
    """
    scales = scales.to(device=layer.weight.device, dtype=layer.weight.dtype)
    with torch.no_grad():
        layer.weight.data.mul_(scales)  # (out, in) *= (in,)

    def _rescale_activation(module: nn.Module, inputs: tuple):
        x = inputs[0]
        return (x / scales,) + inputs[1:]

    layer.register_forward_pre_hook(_rescale_activation)
    layer._fisher_act_scales = scales  # type: ignore[attr-defined]  # for downstream torchao
    return scales


def fisher_scale_model(
    model: nn.Module,
    activations: Optional[Dict[nn.Linear, torch.Tensor]] = None,
    fisher: Optional[Dict[nn.Linear, torch.Tensor]] = None,
    bits: int = 4,
    layer_filter: Optional[Callable[[nn.Linear], bool]] = None,
) -> Dict[str, dict]:
    """Apply C-PTQ Fisher-weighted channel scaling to a model's ``nn.Linear`` layers.

    Model-in, scaled-model-out: walks ``nn.Linear`` modules, computes per-channel
    scales from captured calibration activations (and a Fisher diagonal or its
    activation-energy proxy), folds them in, and returns a small report. Layers
    without supplied activations are left untouched. This is the contract
    ``load_csm_1b(apply_fisher_scaling=True)`` calls into.

    Args:
        model: the model whose Linear layers to scale (e.g. the CSM backbone).
        activations: optional map of ``nn.Linear`` -> (num_samples, in_features)
            calibration inputs, typically captured with forward hooks.
        fisher: optional map of ``nn.Linear`` -> per-channel Fisher diagonal;
            absent entries fall back to the parameter-free activation proxy.
        bits: target quantization width for the simulated objective.
        layer_filter: optional predicate selecting which Linear layers to scale.
    """
    activations = activations or {}
    fisher = fisher or {}

    report: Dict[str, dict] = {}
    layer_count = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if layer_filter is not None and not layer_filter(module):
            continue
        name = f"{type(module).__name__}#{layer_count}"
        layer_count += 1
        layer_acts = activations.get(module)
        if layer_acts is None:
            report[name] = {"scaled": False, "reason": "no calibration activations"}
            continue
        layer_fisher = fisher.get(module)
        if layer_fisher is None:
            layer_fisher = fisher_proxy(layer_acts)
        scales = compute_layer_scales(module.weight.data, layer_acts, layer_fisher, bits=bits)
        apply_layer_scaling(module, scales)
        report[name] = {
            "scaled": True,
            "scale_norm": float(scales.norm().item()),
            "fisher_norm": float(layer_fisher.norm().item()),
        }
    return report
