"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import signal
import warnings
import math
import os
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from filelock import FileLock
from scipy import integrate

from nanochat.common import (
    NANOCHAT_KV_CACHE_TYPE,
    autodetect_device_type,
    compute_init,
    get_base_dir,
)

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
_TURBOQUANT_EPS = 1e-8
_TURBOQUANT_QJL_SCALE = math.sqrt(math.pi / 2.0)
_TURBOQUANT_CODEBOOK_CACHE = {}


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def _fwht(x):
    """Fast Walsh-Hadamard transform over the last dimension."""
    h = 1
    y = x
    while h < y.size(-1):
        y = y.reshape(*y.shape[:-1], -1, 2 * h)
        left = y[..., :h]
        right = y[..., h:]
        y = torch.cat((left + right, left - right), dim=-1)
        y = y.reshape(*x.shape[:-1], -1)
        h *= 2
    return y


def _random_signs(dim, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.int8)
    signs = signs.float().mul_(2.0).sub_(1.0)
    return signs.to(device=device)


def _qr_rotation_matrix(dim, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    rotation = q * diag_sign.unsqueeze(0)
    return rotation.to(device=device)


def _qjl_projection_matrix(dim, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(dim, dim, generator=generator, dtype=torch.float32, device=device)


def _beta_coordinate_pdf(x, dim):
    if x <= -1.0 or x >= 1.0:
        return 0.0
    log_coeff = math.lgamma(dim / 2.0) - 0.5 * math.log(math.pi) - math.lgamma((dim - 1.0) / 2.0)
    return math.exp(log_coeff + ((dim - 3.0) / 2.0) * math.log1p(-(x * x)))


def _solve_beta_lloyd_max(dim, bits, max_iter=200, tol=1e-10):
    """Solve the exact Beta-distribution Lloyd-Max codebook from TurboQuant."""
    n_levels = 1 << bits
    centroids = [(-1.0 + (2.0 * (i + 0.5) / n_levels)) for i in range(n_levels)]
    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [-1.0] + boundaries + [1.0]
        new_centroids = []
        for i in range(n_levels):
            left, right = edges[i], edges[i + 1]
            numerator = integrate.quad(
                lambda x: x * _beta_coordinate_pdf(x, dim),
                left,
                right,
                limit=200,
            )[0]
            denominator = integrate.quad(
                lambda x: _beta_coordinate_pdf(x, dim),
                left,
                right,
                limit=200,
            )[0]
            new_centroids.append(numerator / denominator if denominator > 1e-15 else centroids[i])
        max_shift = max(abs(a - b) for a, b in zip(new_centroids, centroids))
        centroids = new_centroids
        if max_shift < tol:
            break
    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


def _get_codebook_cache_dir():
    cache_dir = f"{get_base_dir()}/turboquant/codebooks"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_turboquant_codebook(dim, bits, device):
    key = (dim, bits)
    if key not in _TURBOQUANT_CODEBOOK_CACHE:
        cache_dir = _get_codebook_cache_dir()
        lock_path = f"{cache_dir}/beta_d{dim}_b{bits}.lock"
        data_path = f"{cache_dir}/beta_d{dim}_b{bits}.pt"
        with FileLock(lock_path):
            if not os.path.exists(data_path):
                centroids, boundaries = _solve_beta_lloyd_max(dim, bits)
                torch.save({"centroids": centroids, "boundaries": boundaries}, data_path)
            payload = torch.load(data_path, map_location="cpu")
        _TURBOQUANT_CODEBOOK_CACHE[key] = (
            payload["centroids"].contiguous(),
            payload["boundaries"].contiguous(),
        )
    centroids, boundaries = _TURBOQUANT_CODEBOOK_CACHE[key]
    return centroids.to(device=device), boundaries.to(device=device)


def _pack_bits(values, bits):
    if bits <= 0:
        raise ValueError("bits must be positive")
    shape = values.shape
    flat = values.reshape(-1, shape[-1]).to(dtype=torch.int32)
    total_bits = shape[-1] * bits
    packed_bytes = (total_bits + 7) // 8
    packed = torch.zeros(flat.size(0), packed_bytes, dtype=torch.int32, device=values.device)
    mask = (1 << bits) - 1
    for idx in range(shape[-1]):
        bit_offset = idx * bits
        byte_idx = bit_offset // 8
        intra = bit_offset % 8
        value = flat[:, idx] & mask
        packed[:, byte_idx] |= (value << intra) & 0xFF
        spill_bits = intra + bits - 8
        if spill_bits > 0:
            packed[:, byte_idx + 1] |= value >> (bits - spill_bits)
    return packed.to(torch.uint8).reshape(*shape[:-1], packed_bytes)


def _unpack_bits(packed, num_values, bits):
    if bits <= 0:
        raise ValueError("bits must be positive")
    shape = packed.shape
    flat = packed.reshape(-1, shape[-1]).to(dtype=torch.int32)
    values = torch.zeros(flat.size(0), num_values, dtype=torch.int32, device=packed.device)
    mask = (1 << bits) - 1
    for idx in range(num_values):
        bit_offset = idx * bits
        byte_idx = bit_offset // 8
        intra = bit_offset % 8
        value = flat[:, byte_idx] >> intra
        spill_bits = intra + bits - 8
        if spill_bits > 0:
            value |= flat[:, byte_idx + 1] << (bits - spill_bits)
        values[:, idx] = value & mask
    return values.to(torch.uint8).reshape(*shape[:-1], num_values)


@dataclass(frozen=True)
class TurboQuantGroup:
    start: int
    end: int
    total_bits: int
    key_mse_bits: int
    value_bits: int
    key_packed_bytes: int
    value_packed_bytes: int

    @property
    def size(self):
        return self.end - self.start


def _build_turboquant_groups(head_dim, kv_cache_type):
    if kv_cache_type == "turbo3":
        layout = [(0, head_dim, 3)]
    elif kv_cache_type == "turbo25":
        split = math.ceil(0.25 * head_dim)
        layout = [(0, split, 3), (split, head_dim, 2)]
    elif kv_cache_type == "turbo35":
        split = math.ceil(0.5 * head_dim)
        layout = [(0, split, 4), (split, head_dim, 3)]
    else:
        raise ValueError(f"Unsupported TurboQuant mode: {kv_cache_type}")
    groups = []
    for start, end, total_bits in layout:
        if end <= start:
            continue
        size = end - start
        key_mse_bits = max(total_bits - 1, 1)
        value_bits = total_bits
        groups.append(TurboQuantGroup(
            start=start,
            end=end,
            total_bits=total_bits,
            key_mse_bits=key_mse_bits,
            value_bits=value_bits,
            key_packed_bytes=(size * key_mse_bits + 7) // 8,
            value_packed_bytes=(size * value_bits + 7) // 8,
        ))
    return groups


class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.is_turbo = False
        self.kv_cache_type = "fp16"
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()


class TurboQuantKVCache:
    """
    KV cache that stores TurboQuant-compressed keys and values.

    This follows TurboQuant (Zandieh et al., 2025): exact Beta Lloyd-Max codebooks,
    random orthogonal rotations, and a 1-bit QJL correction for keys.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype, kv_cache_type):
        if kv_cache_type not in {"turbo3", "turbo25", "turbo35"}:
            raise ValueError(f"Unsupported TurboQuant mode: {kv_cache_type}")
        self.is_turbo = True
        self.kv_cache_type = kv_cache_type
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.groups = _build_turboquant_groups(head_dim, kv_cache_type)
        self.rotation_impl = "fwht" if _is_power_of_two(head_dim) else "qr"
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.prev_embedding = None
        self.qjl_scale = _TURBOQUANT_QJL_SCALE / head_dim

        self.group_codebooks = []
        for group in self.groups:
            key_centroids, key_boundaries = _load_turboquant_codebook(head_dim, group.key_mse_bits, self.device)
            value_centroids, value_boundaries = _load_turboquant_codebook(head_dim, group.value_bits, self.device)
            self.group_codebooks.append({
                "key_centroids": key_centroids,
                "key_boundaries": key_boundaries,
                "value_centroids": value_centroids,
                "value_boundaries": value_boundaries,
            })

        if self.rotation_impl == "fwht":
            self.k_signs = torch.stack(
                [_random_signs(head_dim, 17_003 + layer_idx, self.device) for layer_idx in range(num_layers)],
                dim=0,
            )
            self.v_signs = torch.stack(
                [_random_signs(head_dim, 29_011 + layer_idx, self.device) for layer_idx in range(num_layers)],
                dim=0,
            )
            self.k_rotations = None
            self.v_rotations = None
        else:
            # TODO: experiment with padded FWHT for non-power-of-two head dims in a future fused path.
            self.k_signs = None
            self.v_signs = None
            self.k_rotations = torch.stack(
                [_qr_rotation_matrix(head_dim, 17_003 + layer_idx, self.device) for layer_idx in range(num_layers)],
                dim=0,
            )
            self.v_rotations = torch.stack(
                [_qr_rotation_matrix(head_dim, 29_011 + layer_idx, self.device) for layer_idx in range(num_layers)],
                dim=0,
            )
        self.k_qjl = torch.stack(
            [_qjl_projection_matrix(head_dim, 41_021 + layer_idx, self.device) for layer_idx in range(num_layers)],
            dim=0,
        )

        self.k_code_cache = [
            torch.zeros(
                num_layers, batch_size, seq_len, num_heads, group.key_packed_bytes,
                device=device, dtype=torch.uint8,
            )
            for group in self.groups
        ]
        self.v_code_cache = [
            torch.zeros(
                num_layers, batch_size, seq_len, num_heads, group.value_packed_bytes,
                device=device, dtype=torch.uint8,
            )
            for group in self.groups
        ]
        self.k_qjl_signs = torch.zeros(
            num_layers, batch_size, seq_len, num_heads, (head_dim + 7) // 8,
            device=device, dtype=torch.uint8,
        )
        self.k_vec_norms = torch.zeros(num_layers, batch_size, seq_len, num_heads, device=device, dtype=torch.float16)
        self.k_residual_norms = torch.zeros(num_layers, batch_size, seq_len, num_heads, device=device, dtype=torch.float16)
        self.v_vec_norms = torch.zeros(num_layers, batch_size, seq_len, num_heads, device=device, dtype=torch.float16)

    def reset(self):
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        raise RuntimeError("TurboQuantKVCache does not expose raw FA3 cache tensors")

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens

    def _rotate(self, x, layer_idx, kind):
        if self.rotation_impl == "fwht":
            signs = self.k_signs[layer_idx] if kind == "k" else self.v_signs[layer_idx]
            return _fwht(x * signs) / math.sqrt(self.head_dim)
        rotation = self.k_rotations[layer_idx] if kind == "k" else self.v_rotations[layer_idx]
        return torch.matmul(x, rotation.t())

    def _inverse_rotate(self, x, layer_idx, kind):
        if self.rotation_impl == "fwht":
            signs = self.k_signs[layer_idx] if kind == "k" else self.v_signs[layer_idx]
            return (_fwht(x) / math.sqrt(self.head_dim)) * signs
        rotation = self.k_rotations[layer_idx] if kind == "k" else self.v_rotations[layer_idx]
        return torch.matmul(x, rotation)

    def quantize_and_store(self, layer_idx, k, v, start_pos):
        """
        TurboQuant paper (Zandieh et al., 2025): normalize, rotate, Lloyd-Max quantize,
        and store a 1-bit QJL correction for keys before caching.
        """
        end_pos = start_pos + k.size(1)
        k = k.float()
        v = v.float()
        k_norms = torch.linalg.vector_norm(k, dim=-1, keepdim=True).clamp_min(_TURBOQUANT_EPS)
        v_norms = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(_TURBOQUANT_EPS)
        k_unit = k / k_norms
        v_unit = v / v_norms
        k_rot = self._rotate(k_unit, layer_idx, "k")
        v_rot = self._rotate(v_unit, layer_idx, "v")
        k_mse_rot = torch.zeros_like(k_rot)
        v_mse_rot = torch.zeros_like(v_rot)

        for group_idx, group in enumerate(self.groups):
            codebook = self.group_codebooks[group_idx]
            k_group = k_rot[..., group.start:group.end].contiguous()
            k_indices = torch.bucketize(k_group, codebook["key_boundaries"])
            self.k_code_cache[group_idx][layer_idx, :, start_pos:end_pos] = _pack_bits(k_indices, group.key_mse_bits)
            k_mse_rot[..., group.start:group.end] = codebook["key_centroids"][k_indices.long()]

            v_group = v_rot[..., group.start:group.end].contiguous()
            v_indices = torch.bucketize(v_group, codebook["value_boundaries"])
            self.v_code_cache[group_idx][layer_idx, :, start_pos:end_pos] = _pack_bits(v_indices, group.value_bits)
            v_mse_rot[..., group.start:group.end] = codebook["value_centroids"][v_indices.long()]

        k_mse = self._inverse_rotate(k_mse_rot, layer_idx, "k")
        residual = k_unit - k_mse
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        projected = torch.matmul(residual, self.k_qjl[layer_idx].t())
        sign_bits = (projected >= 0).to(torch.uint8)

        self.k_qjl_signs[layer_idx, :, start_pos:end_pos] = _pack_bits(sign_bits, 1)
        self.k_vec_norms[layer_idx, :, start_pos:end_pos] = k_norms.squeeze(-1).to(torch.float16)
        self.k_residual_norms[layer_idx, :, start_pos:end_pos] = residual_norm.to(torch.float16)
        self.v_vec_norms[layer_idx, :, start_pos:end_pos] = v_norms.squeeze(-1).to(torch.float16)

    def get_dequantized_slice(self, layer_idx, start, end, dtype=None):
        """
        TurboQuant paper (Zandieh et al., 2025): reconstruct the active slice by
        inverse-rotating MSE centroids and adding the QJL residual correction for keys.
        """
        target_dtype = self.dtype if dtype is None else dtype
        seq_len = end - start
        k_rot = torch.zeros(self.batch_size, seq_len, self.n_heads, self.head_dim, device=self.device, dtype=torch.float32)
        v_rot = torch.zeros_like(k_rot)

        for group_idx, group in enumerate(self.groups):
            codebook = self.group_codebooks[group_idx]
            k_indices = _unpack_bits(
                self.k_code_cache[group_idx][layer_idx, :, start:end],
                group.size,
                group.key_mse_bits,
            )
            v_indices = _unpack_bits(
                self.v_code_cache[group_idx][layer_idx, :, start:end],
                group.size,
                group.value_bits,
            )
            k_rot[..., group.start:group.end] = codebook["key_centroids"][k_indices.long()]
            v_rot[..., group.start:group.end] = codebook["value_centroids"][v_indices.long()]

        k_mse = self._inverse_rotate(k_rot, layer_idx, "k")
        qjl_sign_bits = _unpack_bits(self.k_qjl_signs[layer_idx, :, start:end], self.head_dim, 1)
        qjl_signs = qjl_sign_bits.float().mul_(2.0).sub_(1.0)
        qjl = self.qjl_scale * self.k_residual_norms[layer_idx, :, start:end].float().unsqueeze(-1)
        qjl = qjl * torch.matmul(qjl_signs, self.k_qjl[layer_idx])
        k = (k_mse + qjl) * self.k_vec_norms[layer_idx, :, start:end].float().unsqueeze(-1)

        v_mse = self._inverse_rotate(v_rot, layer_idx, "v")
        v = v_mse * self.v_vec_norms[layer_idx, :, start:end].float().unsqueeze(-1)
        return k.to(dtype=target_dtype), v.to(dtype=target_dtype)

    def compression_stats(self):
        active_tokens = self.get_pos()
        active_vectors = self.n_layers * self.batch_size * active_tokens * self.n_heads
        key_bits = active_vectors * sum(group.key_packed_bytes * 8 for group in self.groups)
        value_bits = active_vectors * sum(group.value_packed_bytes * 8 for group in self.groups)
        qjl_bits = active_vectors * (((self.head_dim + 7) // 8) * 8)
        norm_bits = active_vectors * 3 * 16
        compressed_bits = key_bits + value_bits + qjl_bits + norm_bits
        fp16_bits = active_vectors * self.head_dim * 16 * 2
        return {
            "active_tokens": active_tokens,
            "compressed_bits": compressed_bits,
            "fp16_bits": fp16_bits,
            "compression_ratio": fp16_bits / max(compressed_bits, 1),
        }

    def prefill(self, other):
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert isinstance(other, TurboQuantKVCache), "TurboQuant prefill expects another TurboQuant cache"
        assert self.kv_cache_type == other.kv_cache_type
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        for group_idx in range(len(self.groups)):
            source_k = other.k_code_cache[group_idx][:, :, :other_pos]
            source_v = other.v_code_cache[group_idx][:, :, :other_pos]
            self.k_code_cache[group_idx][:, :, :other_pos] = source_k.expand(-1, self.batch_size, -1, -1, -1)
            self.v_code_cache[group_idx][:, :, :other_pos] = source_v.expand(-1, self.batch_size, -1, -1, -1)
        self.k_qjl_signs[:, :, :other_pos] = other.k_qjl_signs[:, :, :other_pos].expand(-1, self.batch_size, -1, -1, -1)
        self.k_vec_norms[:, :, :other_pos] = other.k_vec_norms[:, :, :other_pos].expand(-1, self.batch_size, -1, -1)
        self.k_residual_norms[:, :, :other_pos] = other.k_residual_norms[:, :, :other_pos].expand(-1, self.batch_size, -1, -1)
        self.v_vec_norms[:, :, :other_pos] = other.v_vec_norms[:, :, :other_pos].expand(-1, self.batch_size, -1, -1)
        self.cache_seqlens.fill_(other_pos)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()


def _create_kv_cache(*, kv_cache_type, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
    if kv_cache_type == "fp16":
        return KVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )
    if kv_cache_type in {"turbo3", "turbo25", "turbo35"}:
        return TurboQuantKVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
            kv_cache_type=kv_cache_type,
        )
    raise ValueError(f"Unsupported kv_cache_type: {kv_cache_type}")

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer, kv_cache_type=NANOCHAT_KV_CACHE_TYPE):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use
        self.kv_cache_type = kv_cache_type

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        # NOTE: setting the dtype here and in this way is an ugly hack.
        # Currently the repo assumes that cuda -> bfloat16 and everything else -> float32.
        # We need to know the dtype here to call __init__ on KVCache and pre-allocate its tensors.
        # As a quick hack, we're making generate() function inherit and know about this repo-wise assumption.
        # I think there has to be a bigger refactor to deal with device/dtype tracking across the codebase.
        # In particular, the KVCache should allocate its tensors lazily
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = _create_kv_cache(
            kv_cache_type=self.kv_cache_type,
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = _create_kv_cache(
            kv_cache_type=self.kv_cache_type,
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    from nanochat.checkpoint_manager import load_model
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
