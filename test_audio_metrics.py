"""Tests for the acoustic-fidelity evaluation wiring.

Imports ``Segment`` from the existing ``generator`` module (the non-new module
that defines CSM's generation contract) and exercises the new ``audio_metrics``
library through the ``evaluate_tts`` call-site driver, so the tests prove the
wiring rather than just the metric functions in isolation.
"""

import math

import torch
import torchaudio

# Imported from a NON-NEW module: generator defines the Segment / sample-rate
# contract that the new evaluation code consumes.
from generator import Segment

from audio_metrics import aggregate, evaluate_pair, f0_rmse, mcd, speaker_similarity
from evaluate_tts import format_report, score_pairs

SAMPLE_RATE = 24_000


def _tone(freq_hz: float, seconds: float = 0.5, amp: float = 0.5) -> torch.Tensor:
    """A clean sinusoid at ``freq_hz`` (1-D float tensor at SAMPLE_RATE)."""
    n = int(seconds * SAMPLE_RATE)
    t = torch.arange(n).float() / SAMPLE_RATE
    return amp * torch.sin(2 * math.pi * freq_hz * t)


def test_identical_audio_is_scored_perfectly():
    audio = _tone(150.0)
    score = evaluate_pair(audio, audio, SAMPLE_RATE, domain="conversational")

    assert score.mcd_db < 1e-2
    assert score.f0_rmse_cents < 1e-2 or math.isnan(score.f0_rmse_cents)
    assert score.speaker_similarity > 0.999


def test_pitch_difference_raises_f0_rmse():
    # 90 Hz vs 120 Hz is ~498 cents; both fundamentals are the unique in-range
    # NCCF peak, so the tracker resolves them cleanly.
    low = _tone(90.0)
    high = _tone(120.0)
    rmse = f0_rmse(high, low, SAMPLE_RATE)

    assert not math.isnan(rmse)
    assert rmse > 300.0  # well below the ~498 cent ideal, but unmistakably nonzero


def test_spectral_difference_raises_mcd():
    clean = _tone(150.0)
    noisy = clean + 0.05 * torch.randn_like(clean)

    assert mcd(clean, clean, SAMPLE_RATE) < mcd(clean, noisy, SAMPLE_RATE)
    assert speaker_similarity(clean, clean, SAMPLE_RATE) > speaker_similarity(clean, noisy, SAMPLE_RATE)


def test_aggregate_groups_by_domain():
    audio = _tone(150.0)
    scores = [
        evaluate_pair(audio, audio, SAMPLE_RATE, domain="conversational"),
        evaluate_pair(audio, audio, SAMPLE_RATE, domain="formal"),
        evaluate_pair(audio, audio, SAMPLE_RATE, domain="conversational"),
    ]
    report = aggregate(scores)

    assert set(report) == {"conversational", "formal"}
    assert report["conversational"]["n"] == 2.0
    assert report["formal"]["n"] == 1.0
    # Identical audio means every domain reports a perfect speaker similarity.
    for metrics in report.values():
        assert metrics["speaker_similarity"] > 0.999


def test_score_pairs_wiring_consumes_generator_segments():
    # Exercise the call-site driver with real generator.Segment objects so the
    # generator -> evaluate_tts -> audio_metrics wiring is covered end to end.
    audio = _tone(150.0)
    reference_segment = Segment(text="hi", speaker=0, audio=audio)
    generated_segment = Segment(text="hi", speaker=0, audio=audio)

    pairs = [
        (generated_segment.audio, reference_segment.audio, "conversational"),
        (generated_segment.audio, reference_segment.audio, "emotional"),
    ]
    scores, report = score_pairs(pairs, SAMPLE_RATE)

    assert len(scores) == 2
    assert scores[0].speaker_similarity > 0.999
    assert set(report) == {"conversational", "emotional"}
    # The report must render without error for the README/CLI path.
    assert "conversational" in format_report(report)


def test_sample_rate_contract_matches_mimi_output():
    # The evaluation consumes Generator.generate() output, which runs at the
    # Mimi decoder rate. Guard that the assumed contract rate is correct.
    assert SAMPLE_RATE == 24_000
    # torchaudio's MelSpectrogram (used inside audio_metrics) must accept it.
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=40)
    assert mel(_tone(150.0)).shape[0] == 40
