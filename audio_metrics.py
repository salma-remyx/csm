"""Objective acoustic-quality metrics for CSM-generated speech.

Computes the objective evaluation metrics from the multi-metric TTS
benchmarking framework of "Domain-Specific Evaluation of Text-to-Speech
Systems: A Multi-Metric Benchmarking Study" (arXiv:2608.02235):

* **Mel-cepstral distortion (MCD)** in dB -- spectral-envelope distance.
* **F0 RMSE** in cents -- fundamental-frequency tracking error.

Both metrics compare a generated audio tensor (e.g. the output of
``generator.Generator.generate``) against a ground-truth reference, and
``score_corpus`` aggregates them per speech domain so different styles
(conversational, emotional, ...) can be contrasted -- the paper's
domain-specific analysis applied to this model's own output.

Adapted port (Mode 2). The objective metrics are implemented at full
fidelity with target-native ``torchaudio`` primitives: an orthonormal
DCT-II over a log mel-spectrogram yields the mel-cepstra, and
autocorrelation pitch tracking yields F0. The paper's *subjective*
protocols (MUSHRA, ABX) require human listeners and cannot be automated,
and speaker-similarity scoring via Resemblyzer needs an external
dependency absent from this repo; both are intentionally out of scope
here and belong in a follow-up.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torchaudio

# MCD natural-log -> dB conversion factor: 10 * sqrt(2) / ln(10).
_MCD_SCALE = 10.0 * math.sqrt(2.0) / math.log(10.0)
# One octave = 1200 cents; the cent distance between two pitches is
# 1200 * log2(f1 / f0).
_CENTS_PER_OCTAVE = 1200.0


@dataclass
class MetricResult:
    """Objective acoustic scores for a single generated/reference pair."""

    mcd_db: float
    f0_rmse_cents: float


@dataclass
class DomainScore:
    """Mean objective scores aggregated over one speech domain."""

    domain: str
    n_pairs: int
    mcd_db: float
    f0_rmse_cents: float


def _to_mono(audio: torch.Tensor) -> torch.Tensor:
    """Collapse any leading channel dimension down to a 1-D waveform."""
    if audio.ndim > 1:
        audio = audio.mean(dim=0)
    return audio


def _dct_matrix(n_mels: int, n_mfcc: int) -> torch.Tensor:
    """Orthonormal DCT-II basis, shape ``(n_mfcc, n_mels)``."""
    n = torch.arange(n_mels, dtype=torch.float32)
    k = torch.arange(n_mfcc, dtype=torch.float32).unsqueeze(1)
    matrix = torch.cos(math.pi * (n + 0.5) * k / n_mels)
    matrix = matrix * math.sqrt(2.0 / n_mels)
    matrix[0] = matrix[0] * (1.0 / math.sqrt(2.0))
    return matrix


def _mel_cepstrum(
    audio: torch.Tensor,
    sample_rate: int,
    n_mels: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """Log mel-cepstral coefficients, shape ``(n_mfcc, frames)``."""
    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )(audio)
    log_mel = torch.log(spectrogram + 1e-6)
    basis = _dct_matrix(n_mels, n_mfcc).to(dtype=log_mel.dtype, device=log_mel.device)
    return basis @ log_mel


def mel_cepstral_distortion(
    generated: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
    n_mels: int = 80,
    n_mfcc: int = 25,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """Mel-cepstral distortion (dB) between two audio tensors.

    Standard MCD over cepstral coefficients ``1..M`` (the 0th energy term
    is dropped), converted from the natural-log cepstrum to dB. Returns
    ``nan`` when there are no analysable frames.
    """
    gen_cep = _mel_cepstrum(
        _to_mono(generated).float(), sample_rate, n_mels, n_mfcc, n_fft, hop_length
    )
    ref_cep = _mel_cepstrum(
        _to_mono(reference).float(), sample_rate, n_mels, n_mfcc, n_fft, hop_length
    )
    n_frames = min(gen_cep.size(-1), ref_cep.size(-1))
    if n_frames < 1:
        return float("nan")
    diff = gen_cep[1:, :n_frames] - ref_cep[1:, :n_frames]
    frame_dist = torch.sqrt((diff ** 2).sum(dim=0) + 1e-12)
    return float((_MCD_SCALE * frame_dist.mean()).item())


def _track_f0(
    audio: torch.Tensor,
    sample_rate: int,
    f0_low: int,
    f0_high: int,
    frame_time: float,
) -> torch.Tensor:
    """Per-frame fundamental frequency in Hz; ``0.0`` marks unvoiced frames."""
    return torchaudio.functional.detect_pitch_frequency(
        audio,
        sample_rate,
        frame_time=frame_time,
        freq_low=f0_low,
        freq_high=f0_high,
    )


def f0_rmse_cents(
    generated: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
    f0_low: int = 70,
    f0_high: int = 500,
    frame_time: float = 0.01,
) -> float:
    """RMS fundamental-frequency error in cents over jointly-voiced frames.

    Returns ``0.0`` when the two signals share no voiced frames.
    """
    gen_f0 = _track_f0(_to_mono(generated).float(), sample_rate, f0_low, f0_high, frame_time)
    ref_f0 = _track_f0(_to_mono(reference).float(), sample_rate, f0_low, f0_high, frame_time)
    n = min(gen_f0.numel(), ref_f0.numel())
    voiced = (gen_f0[:n] > 0) & (ref_f0[:n] > 0)
    if not bool(torch.any(voiced)):
        return 0.0
    cents = _CENTS_PER_OCTAVE * torch.log2(gen_f0[:n][voiced] / ref_f0[:n][voiced])
    return float(torch.sqrt((cents ** 2).mean()).item())


def score_pair(
    generated: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
) -> MetricResult:
    """Compute the full objective metric set for one generated/reference pair."""
    return MetricResult(
        mcd_db=mel_cepstral_distortion(generated, reference, sample_rate),
        f0_rmse_cents=f0_rmse_cents(generated, reference, sample_rate),
    )


def score_corpus(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor, str]],
    sample_rate: int,
) -> dict[str, DomainScore]:
    """Aggregate objective scores per speech domain.

    ``pairs`` is a sequence of ``(generated, reference, domain)`` tuples --
    for example with ``domain`` in ``{"conversational", "emotional"}`` --
    mirroring the paper's domain-specific analysis.
    """
    grouped: dict[str, list[MetricResult]] = defaultdict(list)
    for generated, reference, domain in pairs:
        grouped[domain].append(score_pair(generated, reference, sample_rate))

    scores: dict[str, DomainScore] = {}
    for domain, results in grouped.items():
        n_pairs = len(results)
        scores[domain] = DomainScore(
            domain=domain,
            n_pairs=n_pairs,
            mcd_db=sum(r.mcd_db for r in results) / n_pairs,
            f0_rmse_cents=sum(r.f0_rmse_cents for r in results) / n_pairs,
        )
    return scores
