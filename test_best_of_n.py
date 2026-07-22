"""Tests for Best-of-N cross-family ASM rank-ensemble selection.

These cover both the pure ensemble logic in ``best_of_n`` and the wiring of that
logic into the existing call site ``Generator.generate_best_of_n`` in
``generator.py``.

The CSM runtime pulls in heavy, GPU-oriented dependencies (torch, moshi,
transformers, ...) that are not installed in this environment. The integration
test below stubs those modules in ``sys.modules`` before importing ``generator``
so it can exercise the real ``Generator.generate_best_of_n`` method without the
model stack.
"""

import sys
from unittest.mock import MagicMock

# Stub the heavy/external dependencies generator.py imports at module load time.
for _name in [
    "torch",
    "torchaudio",
    "torchaudio.functional",
    "huggingface_hub",
    "moshi",
    "moshi.models",
    "moshi.models.loaders",
    "tokenizers",
    "tokenizers.processors",
    "transformers",
    "models",
    "watermarking",
]:
    sys.modules.setdefault(_name, MagicMock())

import generator  # noqa: E402  -- must follow the sys.modules stubbing above

from best_of_n import (  # noqa: E402
    BestOfNResult,
    conjunctive_max_rank,
    rank_average,
    run_best_of_n,
    select_best_index,
    wer,
)


# Reference text (9 words) and two candidate transcripts per ASR family.
REF = "the quick brown fox jumps over the lazy dog"
# Family A (whisper-like) favours candidate 0; family B (wav2vec2-like) favours 1.
TRANSCRIPTS_A = [
    REF,                                              # 0 edits  -> 0.000
    "the quick brown fox jumps over the lazy",        # 1 del    -> 0.111
    "xxx yyy brown fox jumps over the lazy zzz",      # 3 subs   -> 0.333
]
TRANSCRIPTS_B = [
    "xxx yyy brown fox jumps over the lazy zzz",      # 3 subs   -> 0.333
    "the quick brown fox jumps over the lazy",        # 1 del    -> 0.111
    "xxx yyy brown fox jumps over the lazy dog",      # 2 subs   -> 0.222
]


class _FakeVerifier:
    """A verifier that returns a preset transcript per candidate index."""

    def __init__(self, family, transcripts):
        self.family = family
        self._transcripts = transcripts

    def transcribe(self, audio, sample_rate):
        return self._transcripts[audio]  # audio is the candidate index here


def _make_generator_with_counter(counter):
    """A Generator instance whose ``generate`` returns successive indices."""
    gen = generator.Generator.__new__(generator.Generator)
    gen.sample_rate = 24_000

    def fake_generate(text, speaker, context, max_audio_length_ms, temperature, topk):
        idx = len(counter)
        counter.append(idx)
        return idx

    gen.generate = fake_generate
    return gen


# ---------------------------------------------------------------------------
# Pure ensemble logic (best_of_n module)
# ---------------------------------------------------------------------------


def _wer_matrix():
    return [[wer(REF, t) for t in TRANSCRIPTS_A], [wer(REF, t) for t in TRANSCRIPTS_B]]


def test_wer_basic():
    assert wer("a b c", "a b c") == 0.0
    assert wer("a b c", "a x c") == 1 / 3            # one substitution
    assert wer("a b c", "a b") == 1 / 3              # one deletion
    assert wer("a b c", "a b c d") == 1 / 3          # one insertion
    assert wer("", "anything") == 0.0                # empty reference


def test_rank_average_picks_robust_candidate():
    # ranks within A: [1, 2, 3]; within B: [3, 1, 2]; averages: [2.0, 1.5, 2.5]
    scores = rank_average(_wer_matrix())
    assert scores == [2.0, 1.5, 2.5]
    assert select_best_index(_wer_matrix(), "rank_average") == 1


def test_conjunctive_max_rank_picks_robust_candidate():
    # worst rank per candidate: [3, 2, 3] -> argmin is index 1
    scores = conjunctive_max_rank(_wer_matrix())
    assert scores == [3.0, 2.0, 3.0]
    assert select_best_index(_wer_matrix(), "conjunctive_max_rank") == 1


def test_select_rejects_unknown_ensemble():
    import pytest

    with pytest.raises(ValueError):
        select_best_index(_wer_matrix(), "nope")


# ---------------------------------------------------------------------------
# Integration: Generator.generate_best_of_n (the call-site wiring)
# ---------------------------------------------------------------------------


def test_generate_best_of_n_wiring_rank_average():
    counter = []
    gen = _make_generator_with_counter(counter)
    verifiers = [
        _FakeVerifier("whisper", TRANSCRIPTS_A),
        _FakeVerifier("wav2vec2", TRANSCRIPTS_B),
    ]

    result = gen.generate_best_of_n(
        text=REF, speaker=0, context=[], verifiers=verifiers, n=3, ensemble="rank_average"
    )

    assert isinstance(result, BestOfNResult)
    assert len(counter) == 3                      # generate was called once per candidate
    # Families disagree on the single best candidate -> confound is surfaced.
    assert result.family_top_pick == [0, 1]
    assert result.family_disagreement is True
    # Both ensembles prefer the candidate that is decent in *both* families.
    assert result.selected_index == 1
    assert result.audio == 1                      # candidate at index 1
    assert result.families == ["whisper", "wav2vec2"]
    assert result.n == 3
    assert result.ensemble == "rank_average"


def test_generate_best_of_n_wiring_conjunctive():
    counter = []
    gen = _make_generator_with_counter(counter)
    verifiers = [
        _FakeVerifier("whisper", TRANSCRIPTS_A),
        _FakeVerifier("wav2vec2", TRANSCRIPTS_B),
    ]

    result = gen.generate_best_of_n(
        text=REF, speaker=0, context=[], verifiers=verifiers, n=3,
        ensemble="conjunctive_max_rank",
    )

    assert result.selected_index == 1
    assert result.ensemble == "conjunctive_max_rank"


def test_generate_best_of_n_validates_inputs():
    import pytest

    gen = _make_generator_with_counter([])
    with pytest.raises(ValueError):
        gen.generate_best_of_n(text=REF, speaker=0, context=[], verifiers=[], n=3)
    with pytest.raises(ValueError):
        gen.generate_best_of_n(
            text=REF, speaker=0, context=[], verifiers=[_FakeVerifier("w", TRANSCRIPTS_A)],
            n=0,
        )


def test_run_best_of_n_single_verifier():
    # With one verifier the ensemble reduces to that verifier's own ranking.
    counter = []

    def fake_generate(*args, **kwargs):
        idx = len(counter)
        counter.append(idx)
        return idx

    result = run_best_of_n(
        fake_generate, [_FakeVerifier("whisper", TRANSCRIPTS_A)], REF, 0, [], n=3
    )
    assert result.selected_index == 0            # family A alone prefers candidate 0
    assert result.family_disagreement is False
