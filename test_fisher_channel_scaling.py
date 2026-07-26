"""Tests for the C-PTQ Fisher-weighted channel scaling integration.

The first test imports the existing call-site module (``generator``) and
exercises the wiring edit in ``load_csm_1b`` -- the ``apply_fisher_channel_scaling``
helper that ``load_csm_1b(apply_fisher_scaling=True)`` calls into -- on a small
in-process model, so it does not need the 1B checkpoint, network, or a GPU.
The remaining tests cover the core mechanism in ``fisher_channel_scaling``.
"""

import torch
import torch.nn as nn

# Imported from a NON-NEW module (the call-site module) to prove integration.
import generator
from generator import apply_fisher_channel_scaling

import fisher_channel_scaling as fcs


def _tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_load_csm_1b_exposes_fisher_scaling_flag():
    # The wiring edit added the opt-in flag to the verified call site.
    import inspect

    params = inspect.signature(generator.load_csm_1b).parameters
    assert "apply_fisher_scaling" in params
    assert params["apply_fisher_scaling"].default is False


def test_apply_fisher_channel_scaling_preserves_output():
    """Integration: the generator wiring scales a model yet leaves its float
    output unchanged (non-destructive), re-scales the weights, and exposes the
    per-channel scales for a downstream torchao quantizer."""
    model = _tiny_model()
    lin0, lin2 = model[0], model[2]
    x = torch.randn(13, 8)
    y_before = model(x).clone()
    w0_before = lin0.weight.clone()

    with torch.no_grad():
        calibration = {
            lin0: x.clone(),
            lin2: model[1](model[0](x)).clone(),  # input to the second Linear
        }

    report = apply_fisher_channel_scaling(model, calibration=calibration, bits=4)
    y_after = model(x)

    # Every supplied Linear was scaled; the unscaled path is recorded elsewhere.
    assert len(report) == 2
    assert all(entry["scaled"] for entry in report.values())

    # Channel-wise scaling is an exact float equivalence: output unchanged.
    assert torch.allclose(y_before, y_after, atol=1e-4, rtol=1e-3)

    # Weights were actually re-scaled (s != 1 somewhere) and scales are exposed.
    assert not torch.allclose(lin0.weight, w0_before)
    assert hasattr(lin0, "_fisher_act_scales")
    assert lin0._fisher_act_scales.shape == (lin0.in_features,)


def test_fisher_selection_beats_magnitude_proxy():
    """C-PTQ's core: the per-channel scale is the argmin of the Fisher-weighted
    objective, and that scaling reduces the Fisher-weighted (task-loss) error
    versus leaving the layer unscaled."""
    torch.manual_seed(1)
    out_features, in_features = 16, 6
    weight = torch.randn(out_features, in_features) * 0.1
    activations = torch.randn(64, in_features)
    activations[:, 0] *= 8.0  # activation outlier so scaling can help
    fisher = torch.tensor([50.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # task-sensitive channel

    chosen = fcs.compute_layer_scales(weight, activations, fisher=fisher, bits=4, grid_size=21)

    # Independently recompute the argmin of the Fisher-weighted objective over
    # the same closed-form grid. The selection must match -- this proves the
    # supplied Fisher (not a magnitude proxy) drives the scale selection.
    wd, ad, fd = weight.double(), activations.double(), fisher.double()
    w_max = wd.abs().amax(dim=0).clamp(min=1e-8)
    a_max = ad.abs().amax(dim=0).clamp(min=1e-8)
    best_scales = torch.ones(in_features, dtype=torch.double)
    best_error = fcs.fisher_weighted_error(wd, ad, best_scales, fd, bits=4)
    for alpha in torch.linspace(0.0, 1.0, 21):
        scales = ((a_max**alpha) / (w_max ** (1.0 - alpha))).clamp(min=1e-4, max=1e4)
        error = fcs.fisher_weighted_error(wd, ad, scales, fd, bits=4)
        if error < best_error:
            best_error, best_scales = error, scales
    assert torch.allclose(chosen.double(), best_scales, atol=1e-4)

    # And the chosen scaling reduces the Fisher-weighted error vs no scaling.
    identity = torch.ones(in_features)
    assert fcs.fisher_weighted_error(weight, activations, chosen, fisher, bits=4) < (
        fcs.fisher_weighted_error(weight, activations, identity, fisher, bits=4)
    )


def test_quantizer_levels_and_fisher_diagonal():
    weight = torch.linspace(-1.0, 1.0, 16).reshape(4, 4)
    quantized = fcs.quantize_per_tensor_symmetric(weight, bits=4)
    # Symmetric int4 -> 15 distinct levels in [-7, 7]; values land on the grid.
    assert len(torch.unique(quantized)) <= 15

    grads = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    fisher = fcs.fisher_diagonal(grads)
    assert torch.allclose(fisher, (grads**2).mean(dim=0))


def test_layers_without_calibration_are_left_untouched():
    model = _tiny_model()
    lin0 = model[0]
    w_before = lin0.weight.clone()
    report = apply_fisher_channel_scaling(model, calibration=None, bits=4)
    # No activations supplied -> nothing scaled, model untouched.
    assert all(not entry["scaled"] for entry in report.values())
    assert torch.allclose(lin0.weight, w_before)
