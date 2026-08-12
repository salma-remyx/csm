"""Run domain-specific acoustic evaluation over CSM generations.

Parallels ``run_csm.py``: where ``run_csm.py`` drives generation, this drives
evaluation. It scores ``Generator.generate()`` outputs against same-speaker
reference clips with the objective metrics in ``audio_metrics`` (MCD, F0 RMSE,
speaker similarity) and groups the results by speech domain — the
domain-specific analysis that "Domain-Specific Evaluation of Text-to-Speech
Systems" (arXiv:2608.02235) centers on.

Two entry points:
  * ``score_conversation`` — generate each turn with a CSM ``Generator`` and
    score it against the matching speaker's reference clip.
  * ``score_pairs`` — score precomputed ``(generated, reference, domain)``
    audio pairs. Model-free, so it is unit-testable without a GPU.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

import torch
import torchaudio

from audio_metrics import SegmentScore, aggregate, evaluate_pair
from generator import Segment

# CSM / Mimi output rate. The authoritative source is ``Generator.sample_rate``;
# this constant only covers the WAV-pair CLI path where no generator is loaded.
CSM_SAMPLE_RATE = 24_000

# (text, speaker, domain)
Turn = Tuple[str, int, str]
# (generated_audio, reference_audio, domain)
AudioPair = Tuple[torch.Tensor, torch.Tensor, str]


def score_pairs(pairs: Sequence[AudioPair], sample_rate: int) -> Tuple[List[SegmentScore], Dict[str, Dict[str, float]]]:
    """Score ``(generated, reference, domain)`` audio pairs.

    Returns the per-clip scores and the per-domain aggregate report. Has no
    dependency on the model, so it can be exercised in unit tests.
    """
    scores = [evaluate_pair(gen, ref, sample_rate, domain=domain) for gen, ref, domain in pairs]
    return scores, aggregate(scores)


def score_conversation(
    generator,
    turns: Sequence[Turn],
    speaker_references: Dict[int, Segment],
) -> Tuple[List[SegmentScore], Dict[str, Dict[str, float]]]:
    """Generate each turn and score it against the matching speaker's reference.

    Args:
        generator: a CSM ``Generator`` (see ``load_csm_1b``).
        turns: sequence of ``(text, speaker, domain)`` describing the
            conversation to synthesize and the domain to tag each turn with.
        speaker_references: maps each speaker id to a same-speaker reference
            ``Segment``; each generation is scored against its speaker's
            reference audio.

    Returns:
        ``(scores, report)`` — per-clip ``SegmentScore`` objects and the
        per-domain aggregate report.
    """
    context: List[Segment] = []
    pairs: List[AudioPair] = []
    for text, speaker, domain in turns:
        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=list(context),
            max_audio_length_ms=10_000,
        )
        segment = Segment(text=text, speaker=speaker, audio=audio)
        context.append(segment)
        reference = speaker_references[speaker].audio
        pairs.append((audio, reference, domain))
    return score_pairs(pairs, generator.sample_rate)


def format_report(report: Dict[str, Dict[str, float]]) -> str:
    """Render a per-domain aggregate report as a fixed-width text table."""
    lines = [f"{'domain':<16} {'n':>3} {'MCD(dB)':>9} {'F0-RMSE(c)':>11} {'SpeakerSim':>11}"]
    for domain, metrics in sorted(report.items()):
        lines.append(
            f"{domain:<16} {int(metrics['n']):>3} {metrics['mcd_db']:>9.2f} "
            f"{metrics['f0_rmse_cents']:>11.1f} {metrics['speaker_similarity']:>11.3f}"
        )
    return "\n".join(lines)


def _load_wav(path: str) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(path)
    return torchaudio.functional.resample(audio.squeeze(0), sample_rate, CSM_SAMPLE_RATE)


def main() -> None:
    """Score a generated WAV against a reference WAV (model-free CLI path)."""
    parser = argparse.ArgumentParser(description="Score a generated clip against a reference clip.")
    parser.add_argument("--generated", required=True, help="Path to the generated WAV.")
    parser.add_argument("--reference", required=True, help="Path to the same-speaker reference WAV.")
    parser.add_argument(
        "--domain",
        default="conversational",
        help="Speech domain tag for this pair (e.g. formal, conversational, emotional).",
    )
    args = parser.parse_args()

    generated = _load_wav(args.generated)
    reference = _load_wav(args.reference)
    _, report = score_pairs([(generated, reference, args.domain)], CSM_SAMPLE_RATE)
    print(format_report(report))


if __name__ == "__main__":
    main()
