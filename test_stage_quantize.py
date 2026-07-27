"""Integration tests for stage-aware quantization.

These import from the existing ``models`` module (the call site: the real
CSM ``Model``) and exercise ``stage_quantize`` against it, proving the
capability integrates with the repo's actual model rather than only
self-testing the new module.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")

import stage_quantize
from models import Model, ModelArgs  # non-new module under test


def _tiny_model(dtype=torch.bfloat16) -> Model:
    """A lean CSM ``Model`` (random init, no weights download).

    Tiny vocab/codebook sizes slash the embedding footprint while leaving
    the transformer Linears -- the quantization targets -- identical to the
    real configuration. ``bfloat16`` matches ``load_csm_1b``'s production
    dtype, which torchao's INT4 recipe requires.
    """
    cfg = ModelArgs(
        backbone_flavor="llama-100M",
        decoder_flavor="llama-100M",
        text_vocab_size=64,
        audio_vocab_size=64,
        audio_num_codebooks=2,
    )
    return Model(cfg).to(dtype=dtype)


def test_quantize_model_applies_heterogeneous_recipes_to_both_stages():
    model = _tiny_model()
    report = stage_quantize.quantize_model(model)

    backbone = report["stages"]["backbone"]
    decoder = report["stages"]["decoder"]

    # Both stages are fully quantized under the default recipe map.
    assert backbone["linears_total"] > 0
    assert backbone["linears_quantized"] == backbone["linears_total"]
    assert decoder["linears_quantized"] == decoder["linears_total"]

    # The recipes differ per stage -- the paper's heterogeneous insight.
    assert backbone["recipe"] == stage_quantize.DEFAULT_STAGE_RECIPES["backbone"]
    assert decoder["recipe"] == stage_quantize.DEFAULT_STAGE_RECIPES["decoder"]
    assert backbone["recipe"] != decoder["recipe"]

    # The LM stage (INT4) is compressed more aggressively than the acoustic
    # decoder (INT8), and the overall footprint shrinks.
    assert (
        backbone["weight_compression_ratio_est"]
        > decoder["weight_compression_ratio_est"]
    )
    assert report["weight_compression_ratio_est"] > 1.0


def test_quantized_model_still_generates_a_frame():
    model = _tiny_model()
    stage_quantize.quantize_model(model)

    model.setup_caches(1)
    tokens = torch.randint(0, 64, (1, 4, 3))  # (batch, seq, audio_num_codebooks+1)
    mask = torch.ones(1, 4, 3).bool()
    pos = torch.arange(0, 4).unsqueeze(0)

    with torch.no_grad():
        out = model.generate_frame(tokens, mask, pos, temperature=0.9, topk=10)

    assert out.shape == (1, 2)  # (batch, audio_num_codebooks)


def test_time_forward_returns_positive_seconds():
    weight = torch.zeros(8, 8)
    elapsed = stage_quantize.time_forward(lambda: weight @ weight)
    assert elapsed >= 0.0
