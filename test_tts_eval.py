import numpy as np
import pytest
import torch
import torchaudio

from generator import Segment
from tts_eval import EvalCase
from tts_eval import (
    DEFAULT_CASES,
    f0_contour,
    f0_rmse_cents,
    mel_cepstral_distortion,
    run_benchmark,
    speaker_similarity,
    SAMPLE_RATE,
)


def sine(hz: float, seconds: float = 1.0, amplitude: float = 0.5) -> torch.Tensor:
    t = torch.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return amplitude * torch.sin(2 * np.pi * hz * t)


class TestMetrics:
    def test_mcd_identical_audio_is_zero(self):
        audio = sine(180.0)
        assert mel_cepstral_distortion(audio, audio) == pytest.approx(0.0, abs=1e-6)

    def test_mcd_is_symmetric_and_positive_for_different_audio(self):
        a, b = sine(150.0), sine(320.0)
        d1, d2 = mel_cepstral_distortion(a, b), mel_cepstral_distortion(b, a)
        assert d1 > 0 and d2 > 0
        assert d1 == pytest.approx(d2, abs=1e-3)

    def test_mcd_aligns_different_lengths(self):
        short, long = sine(200.0, 0.4), sine(200.0, 1.2)
        assert mel_cepstral_distortion(short, long) < mel_cepstral_distortion(short, sine(90.0, 1.2))

    def test_f0_contour_tracks_pitch(self):
        f0 = f0_contour(sine(220.0))
        voiced = f0[np.isfinite(f0)]
        assert len(voiced) > len(f0) * 0.8
        assert voiced.mean() == pytest.approx(220.0, rel=0.02)

    def test_f0_contour_marks_silence_unvoiced(self):
        f0 = f0_contour(torch.zeros(SAMPLE_RATE))
        assert np.all(np.isnan(f0))

    def test_f0_rmse_in_cents(self):
        low, high = sine(200.0), sine(400.0)  # one octave = 1200 cents
        assert f0_rmse_cents(high, low) == pytest.approx(1200.0, rel=0.05)

    def test_f0_rmse_identical_is_zero(self):
        audio = sine(190.0)
        assert f0_rmse_cents(audio, audio) == pytest.approx(0.0, abs=1e-6)

    def test_speaker_similarity_prefers_same_pitch(self):
        gen = sine(200.0)
        assert speaker_similarity(gen, sine(200.0)) > speaker_similarity(gen, sine(60.0))


class TestBenchmark:
    @staticmethod
    def fake_generate(text, speaker, context, max_audio_length_ms, **kwargs):
        """Stands in for ``Generator.generate``, honouring its call contract."""
        assert isinstance(context, list)
        assert all(isinstance(s, Segment) for s in context)
        return sine(150.0 + 40 * speaker, seconds=max_audio_length_ms / 10000)

    def test_default_cases_cover_four_domains(self):
        domains = {case.domain for case in DEFAULT_CASES}
        assert domains == {"formal", "conversational", "literary", "emotional"}

    def test_run_benchmark_groups_by_domain(self):
        report = run_benchmark(self.fake_generate, DEFAULT_CASES[:4])
        assert set(report.domains) == {"formal", "conversational"}
        assert report.domains["formal"].n_cases == 3
        assert all(c.duration_s > 0 for c in report.cases)
        # context-free cases carry no acoustic-fidelity metrics
        assert all(c.mcd_db is None for c in report.cases)

    def test_run_benchmark_scores_prompted_cases(self, tmp_path):
        cases = [
            EvalCase("conversational", 0, "Hey there.", prompt="prompt.wav"),
            EvalCase("conversational", 1, "What's up?", prompt="prompt.wav"),
        ]
        reference = sine(150.0)
        torchaudio.save(tmp_path / "prompt.wav", reference.unsqueeze(0), SAMPLE_RATE)

        report = run_benchmark(self.fake_generate, cases, audio_dir=tmp_path)
        scored = [c for c in report.cases if c.mcd_db is not None]
        assert len(scored) == 2
        assert report.domains["conversational"].mean_mcd_db > 0
        assert report.domains["conversational"].mean_speaker_similarity > 0
        # generated audio was written out alongside the prompt
        assert len(list(tmp_path.glob("conversational_*.wav"))) == 2
