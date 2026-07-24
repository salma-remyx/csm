"""Tests for best_of_n (cross-family rank-ensemble BoN selection).

The rank-ensemble and selection math is kept dependency-free, so the pure
tests below run without the repo's GPU stack. The final test wires the
selection through the real ``generator`` module's ``Segment`` contract and is
guarded on ``torch`` being importable.
"""

import pytest

from best_of_n import (
    CONJUNCTIVE_MAX_RANK,
    RANK_AVERAGE,
    bon_generate,
    select_best,
    wer,
)


def test_wer_identical_is_zero():
    assert wer("hello world", "hello world") == 0.0


def test_wer_empty_reference_is_zero():
    assert wer("", "anything goes") == 0.0


def test_wer_substitution_and_deletion():
    # "the cat sat" -> "the bat": 1 substitution + 1 deletion = 2 edits / 3 ref words
    assert wer("the cat sat", "the bat") == pytest.approx(2 / 3)


def test_select_best_single_candidate():
    idx, ranks = select_best([[0.42]])
    assert idx == 0
    assert ranks == [[0.0]]


def test_select_best_rejects_unknown_method():
    with pytest.raises(ValueError):
        select_best([[0.1, 0.2]], "bogus")


def test_rank_average_and_max_rank_diverge():
    # Verifier A ranks c0<c1<c2; verifier B ranks c2<c1<c0.
    scores = [[0.05, 0.10, 0.20], [0.20, 0.10, 0.05]]
    # rank_average: every candidate averages to rank 1.0 -> tie -> earliest (c0).
    assert select_best(scores, RANK_AVERAGE)[0] == 0
    # conjunctive_max_rank: c0 worst-rank 2, c1 worst-rank 1, c2 worst-rank 2 -> c1.
    assert select_best(scores, CONJUNCTIVE_MAX_RANK)[0] == 1


def test_select_best_handles_ties_with_fractional_ranks():
    # Two candidates tie on the only verifier -> they share the fractional
    # rank for positions 0 and 1 (mean 0.5); the earliest still wins.
    idx, ranks = select_best([[0.5, 0.5]])
    assert idx == 0
    assert ranks[0] == [0.5, 0.5]


class _FakeGenerator:
    """Duck-typed stand-in for ``generator.Generator``.

    Returns a distinguishable candidate per call so verifiers can rank them,
    and records how many times ``generate`` was invoked.
    """

    def __init__(self, factory):
        self.factory = factory
        self.calls = 0

    def generate(self, text, speaker, context, max_audio_length_ms=90_000, temperature=0.9, topk=50):
        self.calls += 1
        return self.factory(self.calls)


def test_bon_generate_calls_generate_n_times_and_picks_best():
    # Candidates are ints 1, 2, 3; the verifier prefers id 2 (lowest cost).
    gen = _FakeGenerator(lambda k: k)

    def verifier(audio, text):
        return {1: 0.9, 2: 0.1, 3: 0.5}[audio]

    best, info = bon_generate(gen, "hi", 0, [], n=3, verifiers=[verifier], method=RANK_AVERAGE)
    assert gen.calls == 3
    assert best == 2  # id 2 is the lowest-cost candidate
    assert info["index"] == 1  # id 2 sits at 0-based index 1


def test_bon_generate_accepts_single_verifier_callable():
    gen = _FakeGenerator(lambda k: k)
    best, info = bon_generate(gen, "hi", 0, [], n=2, verifiers=lambda a, t: a)
    # ids 1, 2 -> costs 1, 2 -> lowest cost is id 1 (index 0)
    assert best == 1
    assert info["index"] == 0


def test_bon_generate_requires_verifier():
    with pytest.raises(ValueError):
        bon_generate(_FakeGenerator(lambda k: k), "hi", 0, [], n=2, verifiers=[])


def test_bon_generate_with_generator_segment_contract():
    """Integration with the real ``generator`` module's Segment/audio contract.

    Exercises the wiring end to end: ``bon_generate`` drives a generator's
    ``generate`` and selects via the rank ensemble. Requires ``torch`` (absent
    from this sandbox, present in the repo's CI environment).
    """
    torch = pytest.importorskip("torch")
    from generator import Segment  # non-new module: anchors the integration

    # Candidates are distinct tensors; the verifier prefers the smallest mean.
    gen = _FakeGenerator(lambda k: torch.full((4,), float(k)))

    def verifier(audio, text):
        return float(audio.mean())

    best, info = bon_generate(
        gen, "hello", 0, [], n=3, verifiers=[verifier], method=CONJUNCTIVE_MAX_RANK
    )
    assert torch.is_tensor(best)
    assert best.mean().item() == 1.0
    assert info["index"] == 0
    assert gen.calls == 3

    # The documented context type round-trips unchanged.
    segment = Segment(speaker=0, text="x", audio=torch.zeros(1))
    assert segment.speaker == 0
    assert segment.text == "x"
