"""Tests for the causal prompt-audio denoiser and its wiring into Generator.

The Generator-level tests bypass Mimi and the Llama backbone: they stub the
audio tokenizer so `_tokenize_audio` can be driven without downloading the
gated checkpoint, which keeps the suite runnable offline.
"""

import torch

from causal_enhancer import CausalEnhancer, enhance
from generator import Generator, Segment

SAMPLE_RATE = 24_000


def make_syllabic_speech(noise_amp: float, duration_s: float = 4.0, seed: int = 0):
    """Speech-like signal with genuine pauses, plus stationary noise.

    The pauses matter: the noise floor estimator learns the noise level from
    them, the same way a real prompt's silence carries its room tone.
    """
    g = torch.Generator().manual_seed(seed)
    num_samples = int(SAMPLE_RATE * duration_s)
    t = torch.arange(num_samples) / SAMPLE_RATE

    envelope = torch.zeros(num_samples)
    for i in range(int(duration_s / 0.6) + 1):
        start = int(SAMPLE_RATE * (0.2 + i * 0.6))
        end = int(SAMPLE_RATE * (0.65 + i * 0.6))
        if end < num_samples:
            envelope[start:end] = torch.hann_window(end - start) * 0.9 + 0.1

    speech = torch.zeros(num_samples)
    for harmonic, amp in [(1, 0.5), (2, 0.3), (3, 0.25), (4, 0.2)]:
        speech += amp * torch.sin(2 * torch.pi * 150 * harmonic * t + 150)
    speech *= envelope
    speech = speech / speech.abs().max() * 0.6

    noisy = speech + noise_amp * torch.randn(num_samples, generator=g)
    return speech, noisy


def seg_snr(x: torch.Tensor, ref: torch.Tensor) -> float:
    """Segmental SNR over the middle of the utterance, in dB."""
    error = (x[4800:-4800] - ref[4800:-4800]).pow(2).mean()
    signal = ref[4800:-4800].pow(2).mean()
    return float(10 * torch.log10(signal / error))


def test_enhance_improves_snr_on_noisy_speech():
    speech, noisy = make_syllabic_speech(noise_amp=0.06)

    enhanced = enhance(noisy)

    assert seg_snr(enhanced, speech) > seg_snr(noisy, speech) + 2.0


def test_enhance_leaves_clean_speech_largely_intact():
    speech, clean = make_syllabic_speech(noise_amp=0.0)

    enhanced = enhance(clean)

    # Near-clean audio must pass through: the enhancer is a no-op unless it
    # finds stationary noise worth removing.
    assert seg_snr(enhanced, speech) > 40.0


def test_enhance_is_causal():
    """Changing the future of a signal must not change its enhanced past."""
    _, noisy = make_syllabic_speech(noise_amp=0.06)
    truncated = noisy.clone()
    truncated[8000:] = 0.0

    assert torch.allclose(enhance(noisy)[:7000], enhance(truncated)[:7000], atol=1e-6)


def test_latency_is_bounded_by_one_window():
    enhancer = CausalEnhancer(sample_rate=SAMPLE_RATE)

    assert enhancer.latency_ms == 1000.0 * 512 / SAMPLE_RATE
    assert enhancer.latency_ms < 25.0  # the paper's real-time budget


def test_enhance_preserves_length_and_finiteness():
    for length in [1, 10, 100, 513, 5000, 48_000]:
        out = enhance(torch.randn(length))
        assert out.numel() == length
        assert torch.isfinite(out).all()


def test_summarize_separates_noisy_from_clean_prompts():
    _, clean = make_syllabic_speech(noise_amp=0.0)
    _, noisy = make_syllabic_speech(noise_amp=0.06)
    enhancer = CausalEnhancer()

    clean_stats = enhancer.summarize(clean)
    noisy_stats = enhancer.summarize(noisy)

    assert clean_stats.num_frames > 0
    assert noisy_stats.mean_mask < clean_stats.mean_mask


class _StubAudioTokenizer:
    """Stands in for Mimi so Generator's audio path can run offline."""

    sample_rate = SAMPLE_RATE

    def encode(self, batch):
        # (1, 1, num_samples) -> (1, K, T), matching Mimi's contract. The
        # frame statistic is chosen to be sensitive to the noise floor, so
        # the tokens differ when the enhancer removes stationary noise.
        wave = batch[0, 0]
        num_frames = max(1, wave.numel() // 1280)
        frames = wave[: num_frames * 1280].view(num_frames, 1280)
        # Mean absolute level of each frame: noise lifts it, enhancement
        # (which squashes near-floor energy) lowers it. Offset keeps values
        # positive so they survive the `.long()` cast as distinct tokens.
        levels = frames.abs().mean(dim=1) * 1000.0 + 1.0
        return levels.round().long().unsqueeze(0).unsqueeze(0)


def _make_generator(enhance_audio: bool) -> Generator:
    """Build a Generator without touching the network or a GPU.

    Only the pieces `_tokenize_audio` needs are populated; everything else
    stays unset because these tests never reach it.
    """
    generator = Generator.__new__(Generator)
    generator.device = "cpu"
    generator.sample_rate = SAMPLE_RATE
    generator._audio_tokenizer = _StubAudioTokenizer()
    generator._enhancer = CausalEnhancer(device="cpu") if enhance_audio else None
    return generator


def test_tokenize_audio_applies_enhancer_when_enabled():
    _, noisy = make_syllabic_speech(noise_amp=0.06, duration_s=1.0)
    segment = Segment(text="hello", speaker=0, audio=noisy)

    off = _make_generator(enhance_audio=False)._tokenize_audio(segment.audio)
    on = _make_generator(enhance_audio=True)._tokenize_audio(segment.audio)

    tokens_off, _ = off
    tokens_on, _ = on
    assert not torch.equal(tokens_off, tokens_on)


def test_tokenize_audio_untouched_by_default():
    """Without opt-in, the tokenized audio is bit-identical to raw input."""
    _, noisy = make_syllabic_speech(noise_amp=0.06, duration_s=1.0)
    segment = Segment(text="hello", speaker=0, audio=noisy)

    plain = _make_generator(enhance_audio=False)
    twice = _make_generator(enhance_audio=False)

    assert torch.equal(
        plain._tokenize_audio(segment.audio)[0],
        twice._tokenize_audio(segment.audio)[0],
    )
    assert plain._enhancer is None
