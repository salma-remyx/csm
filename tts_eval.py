"""Domain-specific evaluation of generated speech.

A small, dependency-light evaluation harness that runs a ``Generator``
(the object returned by :func:`generator.load_csm_1b`) over a set of
prompts grouped by speech domain, and reports acoustic-fidelity metrics
per domain.

Metrics:

* ``mcd`` — mel-cepstral distortion (dB), computed on an DTW-aligned
  frame path so generated and reference utterances of different lengths
  are comparable.
* ``f0_rmse`` — root-mean-square error of fundamental frequency,
  expressed in cents so the error is pitch-shift invariant.
* ``speaker_similarity`` — cosine similarity of mean log-mel embeddings
  between the prompt audio and the generated audio (a lightweight
  stand-in for a learned speaker-verification embedding; same signal,
  no extra dependency).

Domain definitions follow the four speech domains used in
"Domain-Specific Evaluation of Text-to-Speech Systems: A Multi-Metric
Benchmarking Study" (arXiv:2608.02235): Formal, Conversational,
Literary/Storytelling, and Emotional.

Example:

    python tts_eval.py --audio-dir eval_audio --output eval_results.json
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

from generator import Generator, Segment

SAMPLE_RATE = 24_000

# Mel-cepstral analysis settings (Mimi operates at 24 kHz / 12.5 Hz frames).
N_FFT = 1024
HOP_LENGTH = 192
WIN_LENGTH = 1024
N_MELS = 80
FMIN, FMAX = 0, SAMPLE_RATE / 2

F0_MIN_HZ, F0_MAX_HZ = 60.0, 500.0
FRAME_MS = 80.0  # CSM generates one audio frame per 80 ms


@dataclass
class EvalCase:
    """One prompt in one speech domain."""

    domain: str
    speaker: int
    text: str
    prompt: Optional[str] = None  # filename of the reference utterance


@dataclass
class CaseResult:
    domain: str
    speaker: int
    text: str
    prompt: Optional[str]
    n_samples: int
    duration_s: float
    mcd_db: Optional[float] = None
    f0_rmse_cents: Optional[float] = None
    speaker_similarity: Optional[float] = None


@dataclass
class DomainSummary:
    n_cases: int
    mean_mcd_db: Optional[float] = None
    mean_f0_rmse_cents: Optional[float] = None
    mean_speaker_similarity: Optional[float] = None


@dataclass
class BenchmarkReport:
    model: str
    cases: List[CaseResult] = field(default_factory=list)
    domains: Dict[str, DomainSummary] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        header = f"{'domain':<14}{'n':>4}{'MCD dB':>10}{'F0 RMSE':>12}{'spk sim':>10}"
        lines = [f"{self.model}", header, "-" * len(header)]
        for domain, s in sorted(self.domains.items()):
            lines.append(
                f"{domain:<14}{s.n_cases:>4}"
                f"{_fmt(s.mean_mcd_db):>10}"
                f"{_fmt(s.mean_f0_rmse_cents):>12}"
                f"{_fmt(s.mean_speaker_similarity):>10}"
            )
        return lines


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Load a mono audio file resampled to ``sample_rate``."""
    audio, sr = torchaudio.load(path)
    if audio.ndim > 1:
        audio = audio.mean(dim=0)
    return torchaudio.functional.resample(audio, orig_freq=sr, new_freq=sample_rate)


def mel_cepstra(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """(n_frames, n_mels) log-mel cepstra, dB scale, with the energy term zeroed.

    Zeroing c0 makes MCD a spectral-shape distance rather than a loudness
    distance, which is what we want when comparing two different renderings
    of the same text.
    """
    if audio.ndim != 1:
        raise ValueError("audio must be 1-D")
    audio = audio.detach().cpu().float()
    if audio.numel() < WIN_LENGTH:
        audio = torch.nn.functional.pad(audio, (0, WIN_LENGTH - audio.numel()))

    spectrogram = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH),
        center=True,
        return_complex=True,
    ).abs() ** 2

    mel_filterbank = torchaudio.functional.melscale_fbanks(
        n_freqs=spectrogram.size(0),
        f_min=FMIN,
        f_max=FMAX,
        n_mels=N_MELS,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="htk",
    )
    mel = torch.log10(spectrogram.T @ mel_filterbank + 1e-10).numpy()  # (T, n_mels)
    cepstra = np.fft.irfft(mel, axis=-1)  # (T, n_mels), MCEP order = n_mels
    cepstra[:, 0] = 0.0
    return cepstra


def mel_cepstral_distortion(
    generated: torch.Tensor, reference: torch.Tensor, sample_rate: int = SAMPLE_RATE
) -> float:
    """DTW-aligned mel-cepstral distortion in dB."""
    gen = mel_cepstra(generated, sample_rate)
    ref = mel_cepstra(reference, sample_rate)
    path = _dtw_path(gen, ref)

    distances = [
        np.linalg.norm(gen[i] - ref[j]) ** 2 for i, j in zip(*path)
    ]
    if not distances:
        return float("nan")
    return float(np.sqrt(np.mean(distances)) * 10.0 / np.sqrt(2.0) / np.log10(np.e))


def _dtw_path(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Monotone DTW alignment between two (T, D) frame sequences."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    cost = np.full((na + 1, nb + 1), np.inf)
    cost[0, 0] = 0.0
    frame_cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1) ** 2
    for i in range(1, na + 1):
        row = frame_cost[i - 1]
        for j in range(1, nb + 1):
            cost[i, j] = row[j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtrack from (na, nb).
    i, j = na, nb
    ii, jj = [i], [j]
    while i > 1 or j > 1:
        candidates = []
        if i > 1:
            candidates.append((cost[i - 1, j], i - 1, j))
        if j > 1:
            candidates.append((cost[i, j - 1], i, j - 1))
        if i > 1 and j > 1:
            candidates.append((cost[i - 1, j - 1], i - 1, j - 1))
        _, i, j = min(candidates)
        ii.append(i)
        jj.append(j)
    ii, jj = np.array(ii[::-1]), np.array(jj[::-1])
    return ii - 1, jj - 1  # drop the 1-based cost-matrix offset


def f0_contour(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Fundamental frequency (Hz) per frame, NaN on unvoiced frames.

    Autocorrelation pitch tracker over the same 80 ms framing CSM generates
    at — coarse, but parameter-free and enough to compare F0 trajectories
    between two renderings.
    """
    audio = audio.detach().cpu().float().numpy()
    frame = int(FRAME_MS / 1000 * sample_rate)
    hop = frame // 2
    lag_min = int(sample_rate / F0_MAX_HZ)
    lag_max = min(int(sample_rate / F0_MIN_HZ), frame - 1)
    if lag_max <= lag_min or len(audio) < frame:
        return np.array([])

    window = np.hanning(frame)
    contours = []
    for start in range(0, len(audio) - frame + 1, hop):
        segment = audio[start : start + frame] * window
        if not np.any(segment):
            contours.append(np.nan)
            continue
        correlation = np.correlate(segment, segment, mode="full")[frame - 1 :]
        correlation = correlation / (correlation[0] + 1e-10)
        peak_lag = lag_min + int(np.argmax(correlation[lag_min : lag_max + 1]))
        if correlation[peak_lag] < 0.5:
            contours.append(np.nan)  # unvoiced
        else:
            contours.append(sample_rate / peak_lag)
    return np.array(contours)


def f0_rmse_cents(
    generated: torch.Tensor, reference: torch.Tensor, sample_rate: int = SAMPLE_RATE
) -> float:
    """RMS error between F0 contours, in cents, on frames where both are voiced."""
    gen = f0_contour(generated, sample_rate)
    ref = f0_contour(reference, sample_rate)
    n = min(len(gen), len(ref))
    if n == 0:
        return float("nan")
    voiced = np.isfinite(gen[:n]) & np.isfinite(ref[:n])
    if voiced.sum() < 2:
        return float("nan")
    cents = 1200.0 * np.log2(gen[:n][voiced] / ref[:n][voiced])
    return float(np.sqrt(np.mean(cents**2)))


def speaker_embedding(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mean log-mel spectrum, a parameter-free stand-in for a speaker encoder."""
    return mel_cepstra(audio, sample_rate).mean(axis=0)


def speaker_similarity(
    generated: torch.Tensor, reference: torch.Tensor, sample_rate: int = SAMPLE_RATE
) -> float:
    """Cosine similarity between speaker embeddings of two utterances."""
    gen = speaker_embedding(generated, sample_rate)
    ref = speaker_embedding(reference, sample_rate)
    denom = np.linalg.norm(gen) * np.linalg.norm(ref)
    if denom == 0:
        return float("nan")
    return float(np.dot(gen, ref) / denom)


# Reference prompts grouped by speech domain, following the four domains of
# arXiv:2608.02235. Each entry pairs a generation prompt with an utterance
# from the same speaker that grounds the voice.
DEFAULT_CASES: List[EvalCase] = [
    EvalCase("formal", 0, "Thank you for joining today's quarterly review."),
    EvalCase("formal", 0, "The committee will convene at nine o'clock sharp."),
    EvalCase("formal", 0, "Please allow me to summarize the key findings."),
    EvalCase("conversational", 0, "Hey, how's it going? Long time no see."),
    EvalCase("conversational", 0, "Oh really? That's pretty wild."),
    EvalCase("conversational", 0, "Yeah, I was thinking the same thing actually."),
    EvalCase("literary", 0, "Once upon a time, in a quiet village by the sea."),
    EvalCase("literary", 0, "The old lighthouse keeper smiled at the horizon."),
    EvalCase("literary", 0, "And so the long winter slowly came to an end."),
    EvalCase("emotional", 0, "I can't believe we finally made it!"),
    EvalCase("emotional", 0, "That was the hardest news I've ever heard."),
    EvalCase("emotional", 0, "You have no idea how much this means to me."),
]


def run_benchmark(
    generate: Callable[..., torch.Tensor],
    cases: Sequence[EvalCase],
    sample_rate: int = SAMPLE_RATE,
    audio_dir: Optional[Path] = None,
    max_audio_length_ms: float = 10_000,
) -> BenchmarkReport:
    """Evaluate ``generate`` (e.g. ``Generator.generate``) across speech domains.

    If a case supplies a ``prompt`` audio file, it is used both as
    conversational context for generation and as the reference for MCD,
    F0 RMSE and speaker similarity. Cases without a prompt are generated
    context-free and reported without acoustic-fidelity metrics.
    """
    report = BenchmarkReport(model=getattr(generate, "__name__", "generate"))
    audio_dir = Path(audio_dir) if audio_dir else None

    for case in cases:
        context: List[Segment] = []
        reference = None
        if case.prompt is not None:
            audio = load_audio(str(audio_dir / case.prompt) if audio_dir else case.prompt, sample_rate)
            context = [Segment(speaker=case.speaker, text=case.text, audio=audio)]
            reference = audio

        generated = generate(
            text=case.text, speaker=case.speaker, context=context,
            max_audio_length_ms=max_audio_length_ms,
        )
        if audio_dir is not None:
            torchaudio.save(
                str(audio_dir / _audio_name(case)),
                generated.unsqueeze(0).cpu(), sample_rate,
            )

        result = CaseResult(
            domain=case.domain, speaker=case.speaker, text=case.text, prompt=case.prompt,
            n_samples=int(generated.numel()),
            duration_s=round(generated.numel() / sample_rate, 3),
        )
        if reference is not None and generated.numel() > 0:
            reference = reference[: generated.numel()] if reference.numel() > generated.numel() else reference
            result.mcd_db = round(mel_cepstral_distortion(generated, reference, sample_rate), 3)
            result.f0_rmse_cents = round(f0_rmse_cents(generated, reference, sample_rate), 3)
            result.speaker_similarity = round(speaker_similarity(generated, reference, sample_rate), 3)
        report.cases.append(result)

    report.domains = summarize_domains(report.cases)
    return report


def _audio_name(case: EvalCase) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in case.text.lower())[:40]
    return f"{case.domain}_{case.speaker}_{slug}.wav"


def summarize_domains(cases: Sequence[CaseResult]) -> Dict[str, DomainSummary]:
    """Per-domain means of every metric that was computed."""
    domains: Dict[str, List[CaseResult]] = {}
    for case in cases:
        domains.setdefault(case.domain, []).append(case)

    summaries = {}
    for domain, group in domains.items():
        summaries[domain] = DomainSummary(
            n_cases=len(group),
            **{
                f"mean_{name}": _mean(getattr(c, name) for c in group)
                for name in ("mcd_db", "f0_rmse_cents", "speaker_similarity")
            },
        )
    return summaries


def _mean(values) -> Optional[float]:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return round(float(np.mean(finite)), 3) if finite else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="cuda", help="device to load CSM-1B on")
    parser.add_argument("--audio-dir", default="eval_audio", help="where generated wavs are written")
    parser.add_argument("--output", default=None, help="write the JSON report here")
    parser.add_argument("--max-audio-length-ms", type=float, default=10_000)
    args = parser.parse_args()

    from generator import load_csm_1b  # deferred: needs HF credentials + GPU

    Path(args.audio_dir).mkdir(parents=True, exist_ok=True)
    generator = load_csm_1b(device=args.model)
    report = run_benchmark(
        generator.generate,
        DEFAULT_CASES,
        sample_rate=generator.sample_rate,
        audio_dir=Path(args.audio_dir),
        max_audio_length_ms=args.max_audio_length_ms,
    )

    for line in report.summary_lines():
        print(line)
    if args.output:
        Path(args.output).write_text(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
