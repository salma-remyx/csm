"""Best-of-N generation with cross-family ASR rank ensembles.

Best-of-N (BoN) inference improves content consistency of zero-shot
text-to-speech by sampling ``N`` candidates and keeping the one an
automatic speech recognition (ASR) verifier scores best against the
target text.

The candidate-selection machinery here implements the cross-family rank
ensembles from:

    "Best-of-N TTS Evaluation is Confounded by ASR Family Alignment"
    (arXiv:2607.08256)

That work shows a verifier's apparent quality depends strongly on which
ASR *family* judges it -- single-family verifiers can reverse candidate
rankings. The proposed remedy is to score every candidate with several
verifiers from *different* ASR families and fuse their rankings with a
cross-family rank ensemble. Two fusion rules are provided:

  * ``rank_average``         -- Borda-style mean rank across families.
  * ``conjunctive_max_rank`` -- minimax over per-family ranks; favours a
                                candidate no single family ranks poorly.

The selection logic is ASR-backend agnostic: pass any collection of
verifier callables ``verifier(audio, text) -> cost`` (lower is better,
e.g. word error rate). A convenience Whisper-based WER verifier is
available via :func:`build_whisper_verifier` (pulls in ``transformers``
and ``torch`` only when called).

:func:`bon_generate` wraps an existing ``Generator.generate`` -- which is
re-entrant, resetting its caches at the start of every call -- and keeps
the ``Segment``-context-in / audio-out contract of
:func:`generator.Generator.generate`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard torch import
    import torch

__all__ = [
    "RANK_AVERAGE",
    "CONJUNCTIVE_MAX_RANK",
    "ENSEMBLE_METHODS",
    "wer",
    "select_best",
    "bon_generate",
    "build_whisper_verifier",
]

# A verifier maps a candidate audio tensor and the target text to a *cost*:
# lower means the candidate matches the text more closely (e.g. WER).
Verifier = Callable[[Any, str], float]

RANK_AVERAGE = "rank_average"
CONJUNCTIVE_MAX_RANK = "conjunctive_max_rank"
ENSEMBLE_METHODS = (RANK_AVERAGE, CONJUNCTIVE_MAX_RANK)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate of ``hypothesis`` against ``reference``.

    Pure-Python Levenshtein over whitespace-separated word tokens; returns
    ``0.0`` for an empty reference. Used as a parameter-free ASR-agreement
    cost so the rank ensemble can be exercised without an ASR backend.
    """
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0
    h = len(hyp)
    prev = list(range(h + 1))
    for i, ref_word in enumerate(ref, start=1):
        cur = [i] + [0] * h
        for j in range(1, h + 1):
            cost = 0 if ref_word == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[h] / len(ref)


def _fractional_ranks(scores: Sequence[float]) -> list[float]:
    """Fractional (average) ranks over ``scores``; ``0.0`` = best (lowest cost).

    Tied candidates share the mean of the positions they span, so exact ties
    cannot bias the downstream ensemble toward an arbitrary ordering.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _combine(ranks_per_verifier: Sequence[Sequence[float]], method: str) -> list[float]:
    """Fuse per-verifier fractional ranks into one rank per candidate."""
    if not ranks_per_verifier:
        return []
    n = len(ranks_per_verifier[0])
    combined = [0.0] * n
    for i in range(n):
        per_candidate = [ranks[i] for ranks in ranks_per_verifier]
        if method == RANK_AVERAGE:
            combined[i] = sum(per_candidate) / len(per_candidate)
        else:  # CONJUNCTIVE_MAX_RANK -- worst rank across families
            combined[i] = max(per_candidate)
    return combined


def select_best(
    scores_by_verifier: Sequence[Sequence[float]],
    method: str = RANK_AVERAGE,
) -> tuple[int, list[list[float]]]:
    """Return ``(index, ranks)`` of the best candidate under ``method``.

    ``scores_by_verifier`` holds one cost list per verifier (each of length
    ``N``, lower = better). ``ranks`` is the per-verifier fractional ranks
    that were fused. The candidate with the lowest fused rank wins; ties are
    broken toward the earliest candidate.
    """
    if method not in ENSEMBLE_METHODS:
        raise ValueError(f"Unknown ensemble method {method!r}; choose from {ENSEMBLE_METHODS}")
    if not scores_by_verifier:
        raise ValueError("select_best requires at least one verifier's scores")
    n = len(scores_by_verifier[0])
    if n == 0:
        raise ValueError("select_best requires at least one candidate")
    ranks = [_fractional_ranks(verifier_scores) for verifier_scores in scores_by_verifier]
    combined = _combine(ranks, method)
    best = min(range(n), key=lambda i: combined[i])
    return best, ranks


def bon_generate(
    generator: Any,
    text: str,
    speaker: int,
    context: list,
    n: int = 5,
    verifiers: Verifier | Sequence[Verifier] = (),
    method: str = RANK_AVERAGE,
    **generate_kwargs: Any,
) -> "tuple[torch.Tensor, dict]":
    """Best-of-N generation with cross-family rank-ensemble selection.

    Samples ``n`` candidates by calling ``generator.generate`` ``n`` times
    (each call resets the model caches, so the candidates are independent),
    scores every candidate with every verifier in ``verifiers``, and keeps
    the candidate that wins under the ``method`` rank ensemble.

    ``verifiers`` is a single callable or a sequence of callables
    ``verifier(audio, text) -> cost`` (lower is better). Pass verifiers from
    *different* ASR families to realize the paper's cross-family
    triangulation; a single verifier reproduces the family-alignment confound
    the paper warns against.

    Returns ``(audio, info)`` where ``info`` holds the chosen ``index``, the
    per-verifier ``ranks``, and the per-verifier raw ``scores``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    verifier_list = [verifiers] if callable(verifiers) else list(verifiers)
    if not verifier_list:
        raise ValueError("bon_generate requires at least one verifier")

    candidates = [
        generator.generate(text, speaker, context, **generate_kwargs) for _ in range(n)
    ]
    scores = [[verifier(audio, text) for audio in candidates] for verifier in verifier_list]
    index, ranks = select_best(scores, method)
    return candidates[index], {"index": index, "ranks": ranks, "scores": scores}


def build_whisper_verifier(
    model_name: str = "openai/whisper-tiny.en",
    device: int | str | None = None,
    source_sample_rate: int = 24_000,
) -> Verifier:
    """Build a WER verifier from a Whisper ASR model.

    Requires ``transformers`` and ``torch`` (imported lazily on first use, so
    importing this module stays lightweight). The returned callable resamples
    the candidate audio to 16 kHz mono, transcribes it, and returns the WER
    against the target ``text``.

    Whisper is a single ASR family. Pair it with verifiers from other families
    (for example wav2vec 2.0 or HuBERT CTC) to realize cross-family
    triangulation in :func:`bon_generate`.
    """
    import torch  # local import: heavy and only needed when this verifier is built
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model_name, device=device)

    def verifier(audio: Any, text: str) -> float:
        wav = torch.as_tensor(audio).float().reshape(-1)
        if source_sample_rate != 16_000:
            import torchaudio

            wav = torchaudio.functional.resample(wav, source_sample_rate, 16_000)
        result = asr({"raw": wav.cpu().numpy(), "sampling_rate": 16_000})
        return wer(text.strip().lower(), str(result["text"]).strip().lower())

    return verifier
