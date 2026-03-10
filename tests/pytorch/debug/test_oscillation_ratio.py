# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for oscillation ratio histogram and EMA computation."""

import pytest
import torch

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformer_engine.debug.features.utils.stats_computation import (
    _unpack_fp4_packed,
    _fp4_bin_histogram,
    _latent_to_fp4_bin_index,
    nvfp4_qw_histogram,
    nvfp4_w_histogram,
    _compute_oscillation_stat,
    OSCILLATION_STAT_NAMES,
)


def _pack_fp4_pair(lo: int, hi: int) -> int:
    """Pack two 4-bit FP4 indices into one byte (TE layout: lo in low nibble)."""
    return (hi << 4) | (lo & 0x0F)


# ---- _unpack_fp4_packed -----------------------------------------------------

class TestUnpackFP4:
    def test_basic_roundtrip(self):
        packed = torch.tensor([[_pack_fp4_pair(0, 1), _pack_fp4_pair(2, 3)]],
                              dtype=torch.uint8, device=_DEVICE)
        unpacked = _unpack_fp4_packed(packed)
        assert unpacked.shape == (1, 4)
        assert unpacked.tolist() == [[0, 1, 2, 3]]

    def test_all_bins(self):
        pairs = [_pack_fp4_pair(i, i + 1) for i in range(0, 16, 2)]
        packed = torch.tensor([pairs], dtype=torch.uint8, device=_DEVICE)
        unpacked = _unpack_fp4_packed(packed)
        assert unpacked.tolist() == [list(range(16))]


# ---- _fp4_bin_histogram -----------------------------------------------------

class TestFP4BinHistogram:
    def test_merge_positive_zero(self):
        """Positive zero (index 0) merged into negative zero (index 8) -> 15 bins."""
        indices = torch.tensor([0, 8], dtype=torch.long, device=_DEVICE)
        hist = _fp4_bin_histogram(indices)
        assert hist.shape == (15,)
        assert hist[7].item() == 2  # both zeros in one bin
        assert hist.sum().item() == 2

    def test_uniform_bin(self):
        indices = torch.full((100,), 2, dtype=torch.long, device=_DEVICE)
        hist = _fp4_bin_histogram(indices)
        assert hist[1].item() == 100  # bin 2 -> position 1 in 15-bin
        assert hist.sum().item() == 100


# ---- _latent_to_fp4_bin_index -----------------------------------------------

class TestLatentToBinIndex:
    @pytest.mark.parametrize("val,expected_bin", [
        (0.0, 0),      # +0
        (0.3, 1),      # -> 0.5
        (0.5, 1),      # exact 0.5
        (1.0, 2),      # exact 1.0
        (1.25, 2),     # midpoint -> lower bin
        (5.5, 7),      # -> 6.0
        (100.0, 7),    # clamps to 6.0
        (-0.3, 9),     # -> -0.5
        (-1.0, 10),    # exact -1.0
        (-6.0, 15),    # exact -6.0
    ])
    def test_known_values(self, val, expected_bin):
        latent = torch.tensor([val], dtype=torch.float32, device=_DEVICE)
        result = _latent_to_fp4_bin_index(latent, torch.device(_DEVICE))
        assert result.item() == expected_bin


# ---- nvfp4_qw_histogram ----------------------------------------------------

class TestNVFP4QWHistogram:
    def test_none_input(self):
        assert nvfp4_qw_histogram(None) is None

    def test_empty_input(self):
        assert nvfp4_qw_histogram(torch.empty(0, dtype=torch.uint8, device=_DEVICE)) is None

    def test_uniform_packed(self):
        packed = torch.full((4, 8), _pack_fp4_pair(2, 2), dtype=torch.uint8, device=_DEVICE)
        hist = nvfp4_qw_histogram(packed)
        assert hist.shape == (15,)
        assert hist[1].item() == 64  # 4*8*2=64 values, all bin 2
        assert hist.sum().item() == 64

    def test_two_bins(self):
        """Half bin-2, half bin-5 -> two peaks in histogram."""
        row_a = torch.full((1, 8), _pack_fp4_pair(2, 2), dtype=torch.uint8, device=_DEVICE)
        row_b = torch.full((1, 8), _pack_fp4_pair(5, 5), dtype=torch.uint8, device=_DEVICE)
        packed = torch.cat([row_a, row_b], dim=0)
        hist = nvfp4_qw_histogram(packed)
        assert hist[1].item() == 16  # bin 2 -> position 1
        assert hist[4].item() == 16  # bin 5 -> position 4
        assert hist.sum().item() == 32


# ---- nvfp4_w_histogram -----------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA for float8 view")
class TestNVFP4WHistogram:
    def test_none_inputs(self):
        assert nvfp4_w_histogram(None, None, None) is None

    def test_identity_scale(self):
        """With scale_inv=1.0 and amax=2688.0, encode_scale=1 -> latent=w."""
        M, K = 4, 16
        w = torch.ones(M, K, dtype=torch.float32, device=_DEVICE)
        scale_inv_uint8 = (
            torch.ones(M, K // 16, dtype=torch.float32, device=_DEVICE)
            .to(torch.float8_e4m3fn).view(torch.uint8)
        )
        amax = torch.tensor([2688.0], dtype=torch.float32, device=_DEVICE)
        hist = nvfp4_w_histogram(w, scale_inv_uint8, amax)
        assert hist[1].item() == M * K  # all map to bin 2 (value 1.0)

    def test_rejects_non_divisible_k(self):
        w = torch.randn(4, 15, dtype=torch.float32, device=_DEVICE)
        scale = torch.ones(4, 1, dtype=torch.uint8, device=_DEVICE)
        amax = torch.tensor([1.0], dtype=torch.float32, device=_DEVICE)
        assert nvfp4_w_histogram(w, scale, amax) is None

    def test_zero_amax(self):
        w = torch.randn(4, 16, dtype=torch.float32, device=_DEVICE)
        scale = torch.ones(4, 1, dtype=torch.uint8, device=_DEVICE)
        amax = torch.tensor([0.0], dtype=torch.float32, device=_DEVICE)
        assert nvfp4_w_histogram(w, scale, amax) is None


# ---- _compute_oscillation_stat ---------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA for float8 view")
class TestComputeOscillationStat:

    def _make_mock(self, M, K, packed_val, scale_val=100.0, amax_val=1.0):
        scale_inv = (
            torch.full((M, K // 16), scale_val, dtype=torch.float32, device=_DEVICE)
            .to(torch.float8_e4m3fn).view(torch.uint8)
        )
        return type("MockNVFP4", (), {
            "_rowwise_data": torch.full((M, K // 2), packed_val,
                                        dtype=torch.uint8, device=_DEVICE),
            "_rowwise_scale_inv": scale_inv,
            "_amax_rowwise": torch.tensor([amax_val], dtype=torch.float32, device=_DEVICE),
        })()

    def _make_aux(self, w, mock, persistent_state, iteration, ema_decay=0.75):
        return {
            "_persistent_state": persistent_state,
            "_nvfp4_quantized_tensor": mock,
            "_ema_decay": ema_decay,
            "_iteration": iteration,
        }

    def test_no_persistent_state(self):
        assert _compute_oscillation_stat(torch.randn(4, 16, device=_DEVICE), {}, "oscillation_ratio") == 0.0

    def test_first_iteration_seeds_ema(self):
        M, K = 4, 16
        ps = {}
        w = torch.randn(M, K, device=_DEVICE)
        aux = self._make_aux(w, self._make_mock(M, K, _pack_fp4_pair(3, 3)), ps, 0)
        assert _compute_oscillation_stat(w, aux, "oscillation_ratio") == 0.0
        assert "_ema_qw" in ps
        assert "_ema_w" in ps
        assert ps["_ema_qw"].shape == (15,)

    def test_second_call_produces_nonzero(self):
        M, K = 4, 16
        ps = {}
        w0 = torch.randn(M, K, device=_DEVICE)
        aux0 = self._make_aux(w0, self._make_mock(M, K, _pack_fp4_pair(2, 2)), ps, 0)
        _compute_oscillation_stat(w0, aux0, "oscillation_ratio")

        w1 = torch.randn(M, K, device=_DEVICE) * 5
        aux1 = self._make_aux(w1, self._make_mock(M, K, _pack_fp4_pair(6, 6)), ps, 1)
        ratio = _compute_oscillation_stat(w1, aux1, "oscillation_ratio")
        assert ratio > 0

    def test_identical_steps_converge_to_zero(self):
        M, K = 4, 16
        ps = {}
        packed_val = _pack_fp4_pair(3, 3)
        w = torch.ones(M, K, device=_DEVICE)
        ratios = []
        for step in range(10):
            aux = self._make_aux(w, self._make_mock(M, K, packed_val), ps, step, ema_decay=0.5)
            r = _compute_oscillation_stat(w, aux, "oscillation_ratio")
            if step > 0:
                ratios.append(r)
        assert all(r == 0.0 for r in ratios)

    def test_cache_shared_across_variants(self):
        M, K = 4, 16
        ps = {}
        w = torch.randn(M, K, device=_DEVICE)
        aux = self._make_aux(w, self._make_mock(M, K, _pack_fp4_pair(3, 3)), ps, 0)
        r = _compute_oscillation_stat(w, aux, "oscillation_ratio")
        q = _compute_oscillation_stat(w, aux, "oscillation_l1distqw")
        d = _compute_oscillation_stat(w, aux, "oscillation_l1distw")
        assert r == 0.0 and q == 0.0 and d == 0.0
        assert ps["_oscillation_cache"]["_iter"] == 0

    def test_ema_decay_zero_maximizes_delta(self):
        M, K = 8, 32
        ps = {}
        w0 = torch.randn(M, K, device=_DEVICE)
        aux0 = self._make_aux(w0, self._make_mock(M, K, _pack_fp4_pair(2, 2)), ps, 0, ema_decay=0.0)
        _compute_oscillation_stat(w0, aux0, "oscillation_ratio")

        w1 = torch.randn(M, K, device=_DEVICE) * 5
        aux1 = self._make_aux(w1, self._make_mock(M, K, _pack_fp4_pair(7, 7)), ps, 1, ema_decay=0.0)
        l1qw = _compute_oscillation_stat(w1, aux1, "oscillation_l1distqw")
        assert l1qw > 0

        w2 = w1.clone()
        aux2 = self._make_aux(w2, self._make_mock(M, K, _pack_fp4_pair(7, 7)), ps, 2, ema_decay=0.0)
        l1qw_2 = _compute_oscillation_stat(w2, aux2, "oscillation_l1distqw")
        assert l1qw_2 == 0.0

    def test_stat_names_constant(self):
        assert OSCILLATION_STAT_NAMES == {"oscillation_ratio", "oscillation_l1distqw", "oscillation_l1distw"}
