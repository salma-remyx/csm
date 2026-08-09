"""Tests for ``audio_metrics`` and its wiring into ``generator.Generator``.

The metric tests exercise the objective scoring directly. The wiring test
drives ``Generator.evaluate_audio`` -- the hook added in ``generator.py`` --
and asserts it delegates to ``audio_metrics.score_pair``. ``generator.py``
imports heavy native dependencies (moshi / torchtune / silentcipher) that
may be unavailable or version-mismatched in a stripped-down test
environment, so those modules are faked purely to let ``generator``'s
module body load; ``evaluate_audio`` touches none of them and uses the
real ``audio_metrics``.
"""

import contextlib
import importlib
import math
import sys
import types

import pytest
import torch

from audio_metrics import (
    DomainScore,
    MetricResult,
    f0_rmse_cents,
    mel_cepstral_distortion,
    score_corpus,
    score_pair,
)

SAMPLE_RATE = 24_000


def _voice(f0_hz: float, duration: float = 1.0) -> torch.Tensor:
    """A voiced-like sawtooth at ``f0_hz`` for deterministic testing."""
    t = torch.arange(int(SAMPLE_RATE * duration)).float() / SAMPLE_RATE
    return 2.0 * ((t * f0_hz) % 1.0) - 1.0


def _load_generator_class():
    """Return ``generator.Generator``, faking heavy deps only if needed.

    A real import is attempted first. If that fails because ``generator``'s
    native dependencies are missing, the modules its body imports at load
    time are stubbed so the class (and the ``evaluate_audio`` hook) can be
    reached. ``evaluate_audio`` never reaches those stubs -- it delegates
    to the real ``audio_metrics.score_pair``.
    """
    with contextlib.suppress(Exception):
        return importlib.import_module("generator").Generator

    stubs = {
        "models": types.ModuleType("models"),
        "watermarking": types.ModuleType("watermarking"),
        "moshi": types.ModuleType("moshi"),
        "moshi.models": types.ModuleType("moshi.models"),
        "moshi.models.loaders": types.ModuleType("moshi.models.loaders"),
        "tokenizers": types.ModuleType("tokenizers"),
        "tokenizers.processors": types.ModuleType("tokenizers.processors"),
        "transformers": types.ModuleType("transformers"),
        "silentcipher": types.ModuleType("silentcipher"),
    }
    stubs["models"].Model = object
    stubs["watermarking"].CSM_1B_GH_WATERMARK = None
    stubs["watermarking"].load_watermarker = lambda *args, **kwargs: None
    stubs["watermarking"].watermark = lambda *args, **kwargs: (None, None)
    stubs["moshi"].models = stubs["moshi.models"]
    stubs["moshi.models"].loaders = stubs["moshi.models.loaders"]
    stubs["moshi.models.loaders"].DEFAULT_REPO = ""
    stubs["moshi.models.loaders"].MIMI_NAME = ""
    stubs["moshi.models.loaders"].get_mimi = lambda *args, **kwargs: None
    stubs["tokenizers"].processors = stubs["tokenizers.processors"]
    stubs["tokenizers.processors"].TemplateProcessing = object
    stubs["transformers"].AutoTokenizer = object
    for name, module in stubs.items():
        sys.modules.setdefault(name, module)
    sys.modules.pop("generator", None)
    with contextlib.suppress(Exception):
        return importlib.import_module("generator").Generator
    return None


_GENERATOR_CLS = _load_generator_class()


def test_identical_audio_scores_zero_distortion():
    reference = _voice(150)
    result = score_pair(reference.clone(), reference, SAMPLE_RATE)
    assert result.mcd_db < 1e-3
    assert result.f0_rmse_cents == 0.0


def test_distortion_increases_with_perturbation():
    torch.manual_seed(0)
    reference = _voice(150)
    clean = mel_cepstral_distortion(reference.clone(), reference, SAMPLE_RATE)
    noisy = mel_cepstral_distortion(
        reference + 0.05 * torch.randn(reference.shape), reference, SAMPLE_RATE
    )
    shifted = mel_cepstral_distortion(_voice(180), reference, SAMPLE_RATE)
    assert clean < noisy < shifted


def test_pitch_shift_is_measured_in_cents():
    # 150 -> 180 Hz is ~316 cents; allow a window for frame-level tracking.
    cents = f0_rmse_cents(_voice(180), _voice(150), SAMPLE_RATE)
    assert 250.0 < cents < 400.0


def test_corpus_groups_and_averages_by_domain():
    reference = _voice(150)
    pairs = [
        (reference.clone(), reference, "conversational"),
        (_voice(180), reference, "conversational"),
        (_voice(200), reference, "emotional"),
    ]
    scores = score_corpus(pairs, SAMPLE_RATE)
    assert set(scores) == {"conversational", "emotional"}
    assert isinstance(scores["conversational"], DomainScore)
    assert scores["conversational"].n_pairs == 2
    assert scores["emotional"].n_pairs == 1
    # Conversational includes an identical pair, so it should beat emotional.
    assert scores["conversational"].mcd_db < scores["emotional"].mcd_db


@pytest.mark.skipif(_GENERATOR_CLS is None, reason="generator module could not be loaded")
def test_generator_evaluate_audio_wiring():
    """``Generator.evaluate_audio`` delegates to ``audio_metrics`` at its sample rate."""
    reference = _voice(150)
    generated = _voice(180)

    class _StubGenerator:
        sample_rate = SAMPLE_RATE

    result = _GENERATOR_CLS.evaluate_audio(_StubGenerator(), generated, reference)
    direct = score_pair(generated, reference, SAMPLE_RATE)
    assert isinstance(result, MetricResult)
    assert math.isclose(result.mcd_db, direct.mcd_db, rel_tol=1e-6)
    assert math.isclose(result.f0_rmse_cents, direct.f0_rmse_cents, rel_tol=1e-6)
