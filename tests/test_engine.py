"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import torch
import torch.nn.functional as F
import pytest
from dataclasses import dataclass

from nanochat.engine import Engine, KVCache, TurboQuantKVCache


# -----------------------------------------------------------------------------
# Mock classes for testing Engine without loading a real model

@dataclass
class MockConfig:
    """Minimal config for Engine tests."""
    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128


class MockModel:
    """
    Mock model that returns uniform logits over the vocab.
    This ensures that with temperature > 0, different samples should
    (with very high probability) produce different tokens.
    """
    def __init__(self, vocab_size=262):  # 256 bytes + 6 special tokens
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        """Return uniform logits so sampling is spread across vocab."""
        B, T = ids.shape
        # With FA3, flash_attn_with_kvcache updates cache in-place and we advance position
        if kv_cache is not None:
            kv_cache.advance(T)
        # Uniform logits -> equal probability for all tokens
        logits = torch.zeros(B, T, self.vocab_size)
        return logits


class ByteTokenizer:
    """
    Simple byte-level tokenizer for testing.
    Tokens 0-255 are raw bytes, 256+ are special tokens.
    """
    def __init__(self):
        # Special tokens start at 256
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))  # bytes 0-255
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        # Filter out special tokens before decoding
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


def make_engine(kv_cache_type="fp16", vocab_size=262):
    return Engine(MockModel(vocab_size=vocab_size), ByteTokenizer(), kv_cache_type=kv_cache_type)

def test_kv_cache_basic():
    """Test basic KVCache functionality for FA3."""
    batch_size = 2
    num_heads = 3
    seq_len = 64
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
        dtype=torch.float32,
    )

    # Check initial state
    assert kv_cache.get_pos() == 0
    assert kv_cache.k_cache.shape == (num_layers, batch_size, seq_len, num_heads, head_dim)
    assert kv_cache.v_cache.shape == (num_layers, batch_size, seq_len, num_heads, head_dim)

    # Test advance
    kv_cache.advance(10)
    assert kv_cache.get_pos() == 10

    kv_cache.advance(5)
    assert kv_cache.get_pos() == 15

    # Test reset
    kv_cache.reset()
    assert kv_cache.get_pos() == 0

    # Test get_layer_cache returns correct views
    k_layer0, v_layer0 = kv_cache.get_layer_cache(0)
    assert k_layer0.shape == (batch_size, seq_len, num_heads, head_dim)
    assert v_layer0.shape == (batch_size, seq_len, num_heads, head_dim)


def test_kv_cache_prefill():
    """Test KVCache.prefill() copies data correctly."""
    batch_size = 1
    num_heads = 4
    head_dim = 8
    num_layers = 2

    # Create source cache and advance it
    src_cache = KVCache(
        batch_size=batch_size, num_heads=num_heads, seq_len=32,
        head_dim=head_dim, num_layers=num_layers, device="cpu", dtype=torch.float32,
    )
    # Write some data to source cache
    src_cache.k_cache[0, 0, :16, :, :] = 1.0
    src_cache.v_cache[0, 0, :16, :, :] = 2.0
    src_cache.advance(16)

    # Create destination cache with larger seq_len
    dst_cache = KVCache(
        batch_size=batch_size, num_heads=num_heads, seq_len=64,
        head_dim=head_dim, num_layers=num_layers, device="cpu", dtype=torch.float32,
    )

    # Prefill
    dst_cache.prefill(src_cache)

    # Check position was copied
    assert dst_cache.get_pos() == 16

    # Check data was copied
    assert (dst_cache.k_cache[0, 0, :16, :, :] == 1.0).all()
    assert (dst_cache.v_cache[0, 0, :16, :, :] == 2.0).all()


@pytest.mark.parametrize("kv_cache_type", ["fp16", "turbo3"])
def test_multi_sample_first_token_diversity(kv_cache_type):
    """
    Test that when generating multiple samples, each sample gets an independently
    sampled first token (not a broadcast of the same token to all rows).

    Previously, the first token after prefill was sampled once and broadcast to all
    rows, causing all samples to start identically. The fix expands the prefill logits
    to num_samples and samples independently for each row.

    With uniform logits over 262 tokens and 16 samples, the probability that all
    samples independently pick the same token is (1/262)^15 ≈ 10^-36. So if they're
    all identical, it indicates tokens are being broadcast instead of independently sampled.
    """
    engine = make_engine(kv_cache_type=kv_cache_type, vocab_size=262)

    # Generate 16 samples with temperature=1.0 (stochastic sampling)
    prompt_tokens = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    num_samples = 16

    # Collect the first generated token from each sample
    first_tokens = []
    gen = engine.generate(
        prompt_tokens,
        num_samples=num_samples,
        max_tokens=1,  # We only need the first token
        temperature=1.0,
        seed=42,
    )
    for token_column, token_masks in gen:
        first_tokens = token_column  # This is the first (and only) yield

    # With uniform distribution and 16 samples, they should NOT all be identical
    # If they are all identical, the bug exists (broadcasting instead of sampling)
    unique_tokens = set(first_tokens)
    assert len(unique_tokens) > 1, (
        f"All {num_samples} samples got the same first token ({first_tokens[0]}). "
        f"With uniform logits, this is statistically impossible (~10^-36 probability) "
        f"unless tokens are being broadcast instead of independently sampled."
    )


@pytest.mark.parametrize("kv_cache_type", ["fp16", "turbo3"])
def test_seed_reproducibility(kv_cache_type):
    """Same seed must produce identical output."""
    engine = make_engine(kv_cache_type=kv_cache_type)
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    for seed in [1, 42, 123, 999]:
        r1, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r2, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r3, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        assert r1 == r2 == r3, "Same seed must produce identical output for the same prompt."


@pytest.mark.parametrize("kv_cache_type", ["fp16", "turbo3"])
def test_temperature_zero_determinism(kv_cache_type):
    """Temperature=0 is deterministic regardless of seed."""
    engine = make_engine(kv_cache_type=kv_cache_type)
    prompt = [261, 72, 101, 108, 108, 111]

    r1, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=1)
    r2, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=42)
    r3, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=123)
    assert r1 == r2 == r3, "Temperature=0 must result in the same output for the same prompt regardless of seed."


@pytest.mark.parametrize("kv_cache_type", ["fp16", "turbo3"])
def test_max_tokens_respected(kv_cache_type):
    """Generation stops at max_tokens limit."""
    engine = make_engine(kv_cache_type=kv_cache_type)
    prompt = [261, 72, 101, 108, 108, 111]

    for max_tokens in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, max_tokens=max_tokens)
        num_generated_tokens = len(results[0]) - len(prompt)
        assert num_generated_tokens <= max_tokens, f"Generated {num_generated_tokens} tokens, expected max_tokens={max_tokens} or less."


@pytest.mark.parametrize("kv_cache_type", ["fp16", "turbo3"])
def test_num_samples_count(kv_cache_type):
    """num_samples=N produces exactly N sequences."""
    engine = make_engine(kv_cache_type=kv_cache_type)
    prompt = [261, 72, 101, 108, 108, 111]

    for num_samples in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, num_samples=num_samples, max_tokens=3)
        assert len(results) == num_samples, f"Expected {num_samples} sequences from {num_samples} samples, got {len(results)}"


@pytest.mark.parametrize("kv_cache_type", ["fp16", "turbo3"])
def test_different_seeds_introduce_variation_when_temperature_nonzero(kv_cache_type):
    """With temperature > 0, different seeds should introduce sampling variation."""
    engine = make_engine(kv_cache_type=kv_cache_type)
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    outputs = set()

    for seed in [1, 42, 123, 999, 1000, 1001, 1002, 1003, 1004, 1005]:
        results, _ = engine.generate_batch(
            prompt,
            temperature=1.0,
            max_tokens=5,
            seed=seed,
        )
        outputs.add(tuple(results[0]))

    # Sanity check: sampling actually introduces variation
    assert len(outputs) > 1, "All seeds produced the same output which is statistically highly improbable."


@pytest.mark.parametrize(
    ("kv_cache_type", "min_ratio", "max_value_mse", "min_key_corr", "max_key_bias"),
    [
        ("turbo3", 4.0, 0.20, 0.90, 0.40),
        ("turbo25", 5.0, 0.35, 0.80, 0.50),
        ("turbo35", 4.0, 0.12, 0.90, 0.15),
    ],
)
def test_turboquant_kvcache_basic(kv_cache_type, min_ratio, max_value_mse, min_key_corr, max_key_bias):
    """Round-trip TurboQuant cache reconstruction should preserve values and key inner products."""
    torch.manual_seed(7)
    B, T, H, D = 2, 24, 3, 64
    cache = TurboQuantKVCache(
        batch_size=B,
        num_heads=H,
        seq_len=64,
        head_dim=D,
        num_layers=2,
        device="cpu",
        dtype=torch.float32,
        kv_cache_type=kv_cache_type,
    )
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    cache.quantize_and_store(layer_idx=0, k=k, v=v, start_pos=0)
    cache.advance(T)
    k_hat, v_hat = cache.get_dequantized_slice(layer_idx=0, start=0, end=T, dtype=torch.float32)

    value_mse = F.mse_loss(v_hat, v).item()
    assert value_mse < max_value_mse, f"{kv_cache_type} value MSE too high: {value_mse:.4f}"

    q = torch.randn(B, T, H, D)
    ref_scores = (q * k).sum(dim=-1)
    approx_scores = (q * k_hat).sum(dim=-1)
    score_bias = (approx_scores - ref_scores).mean().abs().item()
    score_corr = torch.corrcoef(torch.stack([ref_scores.flatten(), approx_scores.flatten()]))[0, 1].item()
    assert score_corr > min_key_corr, (
        f"{kv_cache_type} key score correlation too low: {score_corr:.4f}"
    )
    assert score_bias < max_key_bias, (
        f"{kv_cache_type} key score bias too high: {score_bias:.4f}"
    )

    stats = cache.compression_stats()
    assert stats["compression_ratio"] > min_ratio, (
        f"{kv_cache_type} compression ratio too low: {stats['compression_ratio']:.2f}x"
    )


def test_turboquant_needle():
    """TurboQuant should preserve the top retrieval token on a synthetic long-context needle test."""
    torch.manual_seed(13)
    B, T, H, D = 1, 8192, 2, 64
    needle_idx = 6173

    query = F.normalize(torch.randn(B, 1, H, D), dim=-1) * 6.0
    keys = 0.05 * torch.randn(B, T, H, D)
    values = torch.zeros(B, T, H, D)
    keys[:, needle_idx] = query[:, 0] * 2.0
    values[:, needle_idx, :, 0] = 1.0

    fp16_scores = torch.einsum("bqhd,bthd->bqht", query, keys).squeeze(1)
    fp16_top = fp16_scores.argmax(dim=-1)
    assert torch.equal(fp16_top, torch.full_like(fp16_top, needle_idx))

    turbo_cache = TurboQuantKVCache(
        batch_size=B,
        num_heads=H,
        seq_len=T,
        head_dim=D,
        num_layers=1,
        device="cpu",
        dtype=torch.float32,
        kv_cache_type="turbo3",
    )
    turbo_cache.quantize_and_store(layer_idx=0, k=keys, v=values, start_pos=0)
    turbo_cache.advance(T)
    turbo_keys, _ = turbo_cache.get_dequantized_slice(layer_idx=0, start=0, end=T, dtype=torch.float32)
    turbo_scores = torch.einsum("bqhd,bthd->bqht", query, turbo_keys).squeeze(1)
    turbo_top = turbo_scores.argmax(dim=-1)
    assert torch.equal(turbo_top, fp16_top)
