"""Acoustic-fidelity metrics for text-to-speech output.

Implements the objective evaluation suite from "Domain-Specific Evaluation of
Text-to-Speech Systems: A Multi-Metric Benchmarking Study" (arXiv:2608.02235):
mel-cepstral distortion (MCD), F0 RMSE (in cents), and a speaker-similarity
score, computed between a generated clip and a same-speaker reference clip, and
aggregated per speech domain. The paper's central finding is that TTS quality
varies substantially across speech domains (Formal, Conversational, Literary,
Emotional); ``aggregate`` delivers that domain-specific breakdown.

This module consumes CSM's generation contract directly: ``Generator.generate()``
returns a 1-D audio ``Tensor`` at ``Generator.sample_rate`` (24 kHz), and every
function here scores pairs of such tensors.

Implementation mode — adapted port (Mode 2):
  * MCD and F0 RMSE are implemented at full fidelity as described by the paper.
  * The paper estimates speaker similarity with Resemblyzer, a learned GE2E
    speaker encoder (an external dependency shipping pretrained weights). We
    substitute a parameter-free proxy: cosine similarity between L2-normalized
    mean mel-cepstral embeddings. This keeps the dependency surface to
    ``torch``/``torchaudio`` (already required by CSM) while preserving the
    "same speaker sounds similar" signal that drives the metric.
  * The paper's subjective protocols (MUSHRA listening tests, ABX
    discrimination) require human listeners and are intentionally out of scope
    for an automated library; they belong in a downstream PR.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence

import torch
import torchaudio

# MCD scaling constant for natural-log mel-cepstra (10 * sqrt(2) / ln(10)).
_MCD_SCALE = 10.0 * math.sqrt(2.0) / math.log(10.0)


def _to_1d(audio: torch.Tensor | "numpy.ndarray" | Sequence[float]) -> torch.Tensor:
    """Coerce a (possibly multi-channel) waveform into a 1-D float tensor."""
    tensor = torch.as_tensor(audio).float()
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=0)  # downmix channels, e.g. (C, N) -> (N,)
    return tensor


def _dct_matrix(n: int, device: torch.device) -> torch.Tensor:
    """Orthonormal DCT-II matrix of shape (n, n); rows are output cepstral bins."""
    idx = torch.arange(n, device=device).float()
    k = idx.unsqueeze(1)  # (n, 1) output bin
    nn = idx.unsqueeze(0)  # (1, n) input index
    matrix = torch.cos(math.pi / n * (nn + 0.5) * k)
    scale = torch.full((n,), math.sqrt(2.0 / n), device=device)
    scale[0] = math.sqrt(1.0 / n)
    return matrix * scale.unsqueeze(1)


def _mel_cepstra(
    audio: torch.Tensor,
    sample_rate: int,
    n_mfcc: int = 13,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    spectral_floor: float = 1e-4,
) -> torch.Tensor:
    """Natural-log mel-cepstral coefficients, shape ``(n_mfcc, num_frames)``.

    A relative spectral floor (``spectral_floor * peak``) keeps near-empty mel
    bins from dominating the log, so MCD magnitudes stay in the familiar dB
    range rather than being driven by silent bins jumping off an eps floor.
    """
    x = _to_1d(audio)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    ).to(x.device)
    spec = mel(x)  # (n_mels, num_frames)
    spec = spec.clamp(min=spec.max().clamp(min=1e-12) * spectral_floor)
    log_spec = torch.log(spec)
    dct = _dct_matrix(n_mels, x.device)
    return dct[:n_mfcc, :] @ log_spec  # (n_mfcc, num_frames)


def mcd(
    generated: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
    n_mfcc: int = 13,
    **spectral_kwargs,
) -> float:
    """Mel-cepstral distortion in dB between generated and reference audio.

    Excludes the 0th (energy) coefficient, averages the per-frame L2 distance
    over the time the two clips share, and scales by ``10*sqrt(2)/ln(10)``.
    Identical audio scores 0 dB.
    """
    gen = _mel_cepstra(generated, sample_rate, n_mfcc=n_mfcc, **spectral_kwargs)[1:]
    ref = _mel_cepstra(reference, sample_rate, n_mfcc=n_mfcc, **spectral_kwargs)[1:]
    shared = min(gen.size(1), ref.size(1))
    if shared == 0:
        return float("nan")
    diff = gen[:, :shared] - ref[:, :shared]
    frame_l2 = torch.sqrt((diff * diff).sum(dim=0))
    return float((_MCD_SCALE * frame_l2.mean()).item())


def _track_f0(
    audio: torch.Tensor,
    sample_rate: int,
    frame_length: float = 0.048,
    hop_length: float = 0.010,
    fmin: float = 80.0,
    fmax: float = 400.0,
    voicing_threshold: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame fundamental frequency and voicing flag via NCCF pitch tracking.

    Returns ``(f0, voiced)`` where ``f0`` is in Hz (0.0 where unvoiced) and
    ``voiced`` is a boolean mask, both of length ``num_frames``.
    """
    x = _to_1d(audio)
    win = max(int(frame_length * sample_rate), 2)
    hop = max(int(hop_length * sample_rate), 1)
    lag_min = max(1, int(sample_rate / fmax))
    lag_max = min(int(sample_rate / fmin), win - 1)

    if x.numel() < win or lag_max <= lag_min:
        return torch.zeros(0, device=x.device), torch.zeros(0, dtype=torch.bool, device=x.device)

    frames = x.unfold(0, win, hop)  # (num_frames, win)
    lags = torch.arange(lag_min, lag_max + 1, device=x.device)
    nccf = torch.zeros(frames.size(0), lags.numel(), device=x.device)
    for col, lag in enumerate(range(lag_min, lag_max + 1)):
        left = frames[:, : win - lag]
        right = frames[:, lag:win]
        num = (left * right).sum(dim=1)
        den = torch.sqrt((left * left).sum(dim=1) * (right * right).sum(dim=1) + 1e-12)
        nccf[:, col] = num / den

    best_val, best_idx = nccf.max(dim=1)
    voiced = best_val >= voicing_threshold
    best_lag = lags[best_idx].clamp(min=1).float()
    f0 = torch.where(voiced, float(sample_rate) / best_lag, torch.zeros_like(best_lag))
    return f0, voiced


def f0_rmse(
    generated: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
    **pitch_kwargs,
) -> float:
    """RMSE of fundamental-frequency error in cents over jointly voiced frames.

    ``cents = 1200 * log2(f0_gen / f0_ref)``; the RMSE is taken over frames where
    both clips are voiced. Returns ``nan`` when no frame is jointly voiced.
    """
    f_gen, v_gen = _track_f0(generated, sample_rate, **pitch_kwargs)
    f_ref, v_ref = _track_f0(reference, sample_rate, **pitch_kwargs)
    shared = min(f_gen.numel(), f_ref.numel())
    if shared == 0:
        return float("nan")
    both = v_gen[:shared] & v_ref[:shared]
    if bool(both.sum()) == 0:
        return float("nan")
    gen = f_gen[:shared][both].clamp(min=1.0)
    ref = f_ref[:shared][both].clamp(min=1.0)
    cents = 1200.0 * torch.log2(gen / ref)
    return float(torch.sqrt((cents * cents).mean()).item())


def speaker_similarity(
    generated: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
    n_mfcc: int = 20,
    **spectral_kwargs,
) -> float:
    """Speaker-similarity proxy in [-1, 1]: cosine of mean mel-cepstral embeddings.

    Parameter-free substitute for the paper's Resemblyzer d-vector encoder.
    Identical clips score 1.0.
    """
    gen = _mel_cepstra(generated, sample_rate, n_mfcc=n_mfcc, **spectral_kwargs).mean(dim=1)
    ref = _mel_cepstra(reference, sample_rate, n_mfcc=n_mfcc, **spectral_kwargs).mean(dim=1)
    gen = gen / (gen.norm() + 1e-12)
    ref = ref / (ref.norm() + 1e-12)
    return float((gen * ref).sum().item())


@dataclass
class SegmentScore:
    """Per-clip objective scores tagged with the clip's speech domain."""

    domain: str
    mcd_db: float
    f0_rmse_cents: float
    speaker_similarity: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_pair(
    generated: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
    domain: str = "conversational",
) -> SegmentScore:
    """Score a single generated clip against a reference clip."""
    return SegmentScore(
        domain=domain,
        mcd_db=mcd(generated, reference, sample_rate),
        f0_rmse_cents=f0_rmse(generated, reference, sample_rate),
        speaker_similarity=speaker_similarity(generated, reference, sample_rate),
    )


def _mean_skip_nan(values: Iterable[float]) -> float:
    nums = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not nums:
        return float("nan")
    return sum(nums) / len(nums)


def aggregate(scores: Sequence[SegmentScore]) -> Dict[str, Dict[str, float]]:
    """Group ``SegmentScore`` objects by domain and mean each metric.

    Delivers the paper's domain-specific analysis: which speech domains are
    hardest for the system. NaN values are skipped per metric.
    """
    by_domain: Dict[str, List[SegmentScore]] = defaultdict(list)
    for score in scores:
        by_domain[score.domain].append(score)

    report: Dict[str, Dict[str, float]] = {}
    for domain, group in by_domain.items():
        report[domain] = {
            "n": float(len(group)),
            "mcd_db": _mean_skip_nan(s.mcd_db for s in group),
            "f0_rmse_cents": _mean_skip_nan(s.f0_rmse_cents for s in group),
            "speaker_similarity": _mean_skip_nan(s.speaker_similarity for s in group),
        }
    return report
