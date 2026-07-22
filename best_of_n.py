"""Best-of-N candidate selection with cross-family ASR rank ensembles.

Generate ``N`` TTS candidates from an existing generator and select the most
robust one using an ensemble of ASR-verifier *rankings*, so that the choice does
not depend on a single ASR family.

Adapted from "Best-of-N TTS Evaluation is Confounded by ASR Family Alignment"
(arXiv:2607.08256). That paper shows that a Best-of-N verifier's apparent
quality depends strongly on the ASR family used to judge it, and that two
cross-family rank ensembles -- rank-averaging and conjunctive max-rank --
recover more headroom than any single verifier across independent evaluators.
This module ports those two ensembles and the Best-of-N loop faithfully; the
paper's separate LibriSpeech-PC benchmark / evaluation harness is intentionally
out of scope (it belongs in a downstream evaluation PR).

ASR verifiers are supplied by the caller (e.g. Whisper, wav2vec 2.0, HuBERT) via
the :class:`Verifier` protocol, so the family-alignment confound can be
triangulated across two or more families. A reference :class:`WhisperVerifier`
adapter is included for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate between a reference and a hypothesis transcript.

    Normalized word-level Levenshtein distance:
    ``(substitutions + deletions + insertions) / len(reference words)``.
    Returns ``0.0`` for an empty reference. Lower is better. Comparison is
    case- and punctuation-sensitive; normalize inputs upstream if needed.
    """
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, start=1):
            cost = 0 if rw == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = cur
    return prev[-1] / len(ref)


class Verifier(Protocol):
    """An ASR verifier that transcribes audio and belongs to a ``family``.

    ``family`` distinguishes model lineages (e.g. ``"whisper"``, ``"wav2vec2"``,
    ``"hubert"``) so that selection can triangulate across families rather than
    rely on a single one -- the confound the paper identifies.
    """

    family: str

    def transcribe(self, audio: "torch.Tensor", sample_rate: int) -> str: ...


def _ranks_low_is_best(scores: Sequence[float]) -> List[float]:
    """One-based ranks where ``1`` is the best (lowest) score.

    Ties receive the average of the ranks they span (fractional / "competition"
    ranking), matching how the paper aggregates per-family verifier rankings.
    """
    m = len(scores)
    order = sorted(range(m), key=lambda i: scores[i])
    ranks = [0.0] * m
    i = 0
    while i < m:
        j = i
        while j + 1 < m and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # candidates order[i..j] share positions (i+1)..(j+1) (1-based)
        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def rank_average(wer_matrix: Sequence[Sequence[float]]) -> List[float]:
    """Rank-averaging ensemble (per the paper).

    ``wer_matrix[v][c]`` is candidate ``c``'s WER under verifier family ``v``.
    Within each family candidates are ranked (``1`` = best, lower WER is
    better), then ranks are averaged across families. Returns the mean rank per
    candidate; lower is better. Averaging per-family *ranks* (rather than raw
    WER) is what makes this robust to the family-specific score offsets that
    constitute the ASR-family confound.
    """
    if not wer_matrix:
        return []
    n_candidates = len(wer_matrix[0])
    acc = [0.0] * n_candidates
    for family_scores in wer_matrix:
        ranks = _ranks_low_is_best(family_scores)
        for c in range(n_candidates):
            acc[c] += ranks[c]
    return [a / len(wer_matrix) for a in acc]


def conjunctive_max_rank(wer_matrix: Sequence[Sequence[float]]) -> List[float]:
    """Conjunctive max-rank ensemble (per the paper).

    For each candidate take its worst (maximum) per-family rank, then prefer the
    candidate with the best (lowest) worst-case rank. A candidate that any single
    family ranks poorly is penalized, so the selection avoids family-specific
    failures. Returns the worst rank per candidate; lower is better.
    """
    if not wer_matrix:
        return []
    n_candidates = len(wer_matrix[0])
    worst = [0.0] * n_candidates
    for family_scores in wer_matrix:
        ranks = _ranks_low_is_best(family_scores)
        for c in range(n_candidates):
            if ranks[c] > worst[c]:
                worst[c] = ranks[c]
    return worst


_ENSEMBLES = {
    "rank_average": rank_average,
    "conjunctive_max_rank": conjunctive_max_rank,
}


def ensemble_scores(
    wer_matrix: Sequence[Sequence[float]], method: str = "rank_average"
) -> List[float]:
    """Aggregate per-family WERs into a single score per candidate (lower=better)."""
    if method not in _ENSEMBLES:
        raise ValueError(
            f"Unknown ensemble method {method!r}; expected one of {sorted(_ENSEMBLES)}"
        )
    return _ENSEMBLES[method](wer_matrix)


def select_best_index(
    wer_matrix: Sequence[Sequence[float]], method: str = "rank_average"
) -> int:
    """Return the candidate index preferred by the chosen ensemble."""
    scores = ensemble_scores(wer_matrix, method)
    best = 0
    for c in range(1, len(scores)):
        if scores[c] < scores[best]:
            best = c
    return best


@dataclass
class BestOfNResult:
    """Outcome of a Best-of-N selection over multiple ASR-verifier families."""

    audio: Any
    selected_index: int
    ensemble: str
    n: int
    families: List[str] = field(default_factory=list)
    # wer_matrix[family_idx][candidate_idx]; lower is better
    wer_matrix: List[List[float]] = field(default_factory=list)
    # per family, the candidate that family would pick alone (argmin WER) --
    # disagreement here is the ASR-family confound the paper describes.
    family_top_pick: List[int] = field(default_factory=list)

    @property
    def family_disagreement(self) -> bool:
        """True when the verifier families disagree on the single best candidate."""
        return len(set(self.family_top_pick)) > 1


# A generate callable with the Generator.generate signature.
GenerateFn = Callable[..., Any]


def run_best_of_n(
    generate_fn: GenerateFn,
    verifiers: Sequence[Verifier],
    text: str,
    speaker: int,
    context: Sequence[Any],
    n: int = 5,
    max_audio_length_ms: float = 90_000,
    temperature: float = 0.9,
    topk: int = 50,
    ensemble: str = "rank_average",
    sample_rate: int = 24_000,
) -> BestOfNResult:
    """Run Best-of-N selection with a cross-family ASR rank ensemble.

    Calls ``generate_fn(text, speaker, context, max_audio_length_ms,
    temperature, topk)`` ``n`` times, transcribes each candidate with every
    verifier, computes per-family WER against ``text``, and returns the
    ensemble-selected candidate plus the per-family rankings (which surface the
    family-alignment confound when families disagree).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not verifiers:
        raise ValueError("at least one verifier is required")

    candidates = [
        generate_fn(text, speaker, context, max_audio_length_ms, temperature, topk)
        for _ in range(n)
    ]

    wer_matrix: List[List[float]] = []
    families: List[str] = []
    family_top_pick: List[int] = []
    for verifier in verifiers:
        wers = [wer(text, verifier.transcribe(cand, sample_rate)) for cand in candidates]
        wer_matrix.append(wers)
        families.append(getattr(verifier, "family", type(verifier).__name__))
        top = 0
        for c in range(1, len(wers)):
            if wers[c] < wers[top]:
                top = c
        family_top_pick.append(top)

    selected = select_best_index(wer_matrix, ensemble)
    return BestOfNResult(
        audio=candidates[selected],
        selected_index=selected,
        ensemble=ensemble,
        n=n,
        families=families,
        wer_matrix=wer_matrix,
        family_top_pick=family_top_pick,
    )


class WhisperVerifier:
    """Reference ASR verifier wrapping a Whisper model via ``transformers``.

    Whisper is one of the three ASR families studied in the paper (alongside
    wav2vec 2.0 and HuBERT). Supply two or more verifiers from *different*
    families to :func:`run_best_of_n` to triangulate the family-alignment
    confound -- a wav2vec 2.0 / HuBERT verifier follows the same
    :class:`Verifier` protocol. The model is loaded lazily on construction, so
    importing this module never triggers a network download.
    """

    def __init__(self, model: str = "openai/whisper-tiny", device: str = "cpu"):
        from transformers import pipeline

        self.family = "whisper"
        self._pipe = pipeline(
            "automatic-speech-recognition", model=model, device=device
        )

    def transcribe(self, audio: "torch.Tensor", sample_rate: int) -> str:
        import numpy as np

        wav = np.asarray(audio.detach().cpu().float())
        result = self._pipe({"raw": wav, "sampling_rate": sample_rate})
        return result.get("text", "").strip()
