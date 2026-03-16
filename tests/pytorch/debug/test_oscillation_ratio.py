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
    _master_w_bin_histogram,
    _latent_to_master_bin_index,
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


# ---- _master_w_bin_histogram ------------------------------------------------

class TestMasterWBinHistogram:
    def test_merge_positive_zero(self):
        """Positive zero (index 0) merged into negative zero (index 15) -> 29 bins."""
        indices = torch.tensor([0, 15], dtype=torch.long, device=_DEVICE)
        hist = _master_w_bin_histogram(indices)
        assert hist.shape == (29,)
        assert hist[14].item() == 2  # both zeros merged
        assert hist.sum().item() == 2

    def test_bin_count(self):
        indices = torch.full((100,), 4, dtype=torch.long, device=_DEVICE)
        hist = _master_w_bin_histogram(indices)
        assert hist.shape == (29,)
        assert hist[3].item() == 100  # index 4 -> position 3 in 29-bin
        assert hist.sum().item() == 100


# ---- _latent_to_master_bin_index -------------------------------------------

class TestLatentToMasterBinIndex:
    @pytest.mark.parametrize("val,expected_bin", [
        (0.0, 0),       # +0
        (0.1, 0),       # near +0, below 0.125 boundary
        (0.25, 1),      # FP4 boundary → master bin center
        (0.5, 2),       # FP4 value → master bin center
        (0.74, 3),      # near boundary 0.75, below it
        (0.75, 3),      # exactly at FP4 boundary → master bin center
        (0.76, 3),      # near boundary 0.75, above it
        (1.0, 4),       # FP4 value
        (6.0, 14),      # max FP4 value
        (100.0, 14),    # clamps to last bin
        (-0.5, 17),     # negative FP4 value (2 + 15 = 17)
        (-0.75, 18),    # negative FP4 boundary (3 + 15 = 18)
        (-6.0, 29),     # max negative (14 + 15 = 29)
    ])
    def test_known_values(self, val, expected_bin):
        latent = torch.tensor([val], dtype=torch.float32, device=_DEVICE)
        result = _latent_to_master_bin_index(latent, torch.device(_DEVICE))
        assert result.item() == expected_bin


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
        assert hist.shape == (29,)
        # latent=1.0 → master bin 4 (center at 1.0) → position 3 in 29-bin hist
        assert hist[3].item() == M * K

    def test_ignores_nonfinite_latents(self):
        w = torch.tensor(
            [[
                0.25, float("nan"), float("inf"), float("-inf"),
                -0.25, 0.75, -0.75, 6.0,
                0.0, 0.1, 0.5, -0.5,
                1.0, -1.0, 3.5, -3.5,
            ]],
            dtype=torch.float32,
            device=_DEVICE,
        )
        scale_inv_uint8 = (
            torch.ones(1, 1, dtype=torch.float32, device=_DEVICE)
            .to(torch.float8_e4m3fn).view(torch.uint8)
        )
        amax = torch.tensor([2688.0], dtype=torch.float32, device=_DEVICE)
        hist = nvfp4_w_histogram(w, scale_inv_uint8, amax)
        assert hist.shape == (29,)
        assert hist.sum().item() == 13  # NaN/+Inf/-Inf are dropped
        assert hist[0].item() == 1  # +0.25
        assert hist[13].item() == 1  # +6.0, not +Inf
        assert hist[14].item() == 2  # +0.0 and +0.1 merge into zero bin
        assert hist[28].item() == 0  # -Inf is ignored, not clamped to -6.0

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
        assert ps["_ema_w"].shape == (29,)

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
        rr = _compute_oscillation_stat(w, aux, "oscillation_ratio_reduced")
        q = _compute_oscillation_stat(w, aux, "oscillation_l1distqw")
        d = _compute_oscillation_stat(w, aux, "oscillation_l1distw")
        dr = _compute_oscillation_stat(w, aux, "oscillation_l1distw_reduced")
        assert r == 0.0 and rr == 0.0 and q == 0.0 and d == 0.0 and dr == 0.0
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

    def test_boundary_zero_denominator_reports_zero_ratio(self):
        """Kitchen-compatible behavior: zero denominator reports zero ratio.

        With 29-bin master histogram (quarter-point boundaries), small
        oscillations near an FP4 quantization boundary (e.g. 0.75) stay
        within one master bin, so l1distw ≈ 0 while l1distqw > 0.

        Kitchen still reports ratio 0 in this exact case because the
        denominator is zero and the runtime metric guards that path.

        Uses amax=2688.0, scale_inv=1.0 so encode_scale=1 and latent=w.
        """
        M, K = 4, 16

        # Step 0 — seed EMAs.  Quantized = all bin 1 (value 0.5).
        # Master weights at 0.74 → latent=0.74, in master bin 3 [0.625, 0.875).
        ps = {}
        w0 = torch.full((M, K), 0.74, dtype=torch.float32, device=_DEVICE)
        mock0 = self._make_mock(M, K, _pack_fp4_pair(1, 1),
                                scale_val=1.0, amax_val=2688.0)
        aux0 = self._make_aux(w0, mock0, ps, 0, ema_decay=0.0)
        _compute_oscillation_stat(w0, aux0, "oscillation_ratio")

        # Step 1 — master weight moves slightly to 0.76 → latent=0.76,
        # still in master bin 3 [0.625, 0.875).  But the quantized weight
        # flips from FP4 bin 1 (0.5) to bin 2 (1.0).
        w1 = torch.full((M, K), 0.76, dtype=torch.float32, device=_DEVICE)
        mock1 = self._make_mock(M, K, _pack_fp4_pair(2, 2),
                                scale_val=1.0, amax_val=2688.0)
        aux1 = self._make_aux(w1, mock1, ps, 1, ema_decay=0.0)
        ratio = _compute_oscillation_stat(w1, aux1, "oscillation_ratio")
        ratio_reduced = _compute_oscillation_stat(w1, aux1, "oscillation_ratio_reduced")
        l1distqw = _compute_oscillation_stat(w1, aux1, "oscillation_l1distqw")
        l1distw = _compute_oscillation_stat(w1, aux1, "oscillation_l1distw")
        l1distw_reduced = _compute_oscillation_stat(w1, aux1, "oscillation_l1distw_reduced")

        # Master histogram didn't change → l1distw ≈ 0.
        assert l1distw < 1e-6, f"Expected l1distw ≈ 0, got {l1distw}"
        assert l1distw_reduced < 1e-6, f"Expected l1distw_reduced ≈ 0, got {l1distw_reduced}"
        # Quantized histogram changed → l1distqw > 0.
        assert l1distqw > 0.1, f"Expected l1distqw > 0.1, got {l1distqw}"
        # Runtime metric follows kitchen and reports 0 when denominator is exactly 0.
        assert ratio == 0.0
        assert ratio_reduced == 0.0

    def test_old_15bin_degenerate(self):
        """Old 15-bin approach is structurally degenerate: ratio is always exactly 1.

        Since qw = quantize(w), mapping both back to 15 FP4 bins produces
        IDENTICAL histograms at every step.  Therefore l1distqw == l1distw
        and ratio == 1 regardless of whether oscillation is happening.
        The old metric measured nothing useful.
        """
        N = 64
        dev = torch.device(_DEVICE)

        latent_step0 = torch.full((N,), 0.74, dtype=torch.float32, device=_DEVICE)
        latent_step1 = torch.full((N,), 0.76, dtype=torch.float32, device=_DEVICE)

        # qw histogram: comes from actual FP4 quantization.
        # 0.74 quantizes to FP4 bin 1 (value 0.5), 0.76 to bin 2 (value 1.0).
        qw_idx0 = _latent_to_fp4_bin_index(latent_step0, dev)
        qw_idx1 = _latent_to_fp4_bin_index(latent_step1, dev)

        # Old w histogram: master weights mapped to the SAME 15 FP4 bins.
        old_w_idx0 = _latent_to_fp4_bin_index(latent_step0, dev)
        old_w_idx1 = _latent_to_fp4_bin_index(latent_step1, dev)

        # They are IDENTICAL because qw = quantize(w) uses the same bin edges.
        assert torch.equal(qw_idx0, old_w_idx0), "qw and old-w should be in same bins"
        assert torch.equal(qw_idx1, old_w_idx1), "qw and old-w should be in same bins"

        # Therefore histograms match, L1 deltas match, ratio ≡ 1.
        qw_hist0 = _fp4_bin_histogram(qw_idx0).float()
        qw_hist1 = _fp4_bin_histogram(qw_idx1).float()
        old_w_hist0 = _fp4_bin_histogram(old_w_idx0).float()
        old_w_hist1 = _fp4_bin_histogram(old_w_idx1).float()
        assert torch.equal(qw_hist0, old_w_hist0)
        assert torch.equal(qw_hist1, old_w_hist1)

        l1distqw = ((qw_hist1 / N) - (qw_hist0 / N)).abs().sum().item()
        old_l1distw = ((old_w_hist1 / N) - (old_w_hist0 / N)).abs().sum().item()
        assert l1distqw == old_l1distw, "Old approach: L1 deltas are always identical"

        old_ratio = l1distqw / (old_l1distw + 1e-12)
        assert abs(old_ratio - 1.0) < 1e-6, (
            f"Old 15-bin ratio is structurally ≡ 1 (degenerate), got {old_ratio}"
        )

    def test_new_29bin_breaks_degeneracy(self):
        """New 29-bin approach: master bins are finer than FP4 → ratio ≠ 1.

        With quarter-point boundaries, the master histogram captures
        sub-FP4-bin resolution.  Weights that cross an FP4 boundary
        (causing qw to flip) can stay in the same finer master bin,
        breaking the qw==w degeneracy and producing ratio >> 1.
        """
        N = 64
        dev = torch.device(_DEVICE)

        latent_step0 = torch.full((N,), 0.74, dtype=torch.float32, device=_DEVICE)
        latent_step1 = torch.full((N,), 0.76, dtype=torch.float32, device=_DEVICE)

        # qw histogram (15-bin FP4): flips from bin 1 to bin 2.
        qw_hist0 = _fp4_bin_histogram(
            _latent_to_fp4_bin_index(latent_step0, dev)
        ).float()
        qw_hist1 = _fp4_bin_histogram(
            _latent_to_fp4_bin_index(latent_step1, dev)
        ).float()
        l1distqw = ((qw_hist1 / N) - (qw_hist0 / N)).abs().sum().item()
        assert l1distqw > 1.0, f"Quant histogram should change a lot, got {l1distqw}"

        # New w histogram (29-bin): both 0.74 and 0.76 stay in master bin 3.
        new_w_hist0 = _master_w_bin_histogram(
            _latent_to_master_bin_index(latent_step0, dev)
        ).float()
        new_w_hist1 = _master_w_bin_histogram(
            _latent_to_master_bin_index(latent_step1, dev)
        ).float()
        new_l1distw = ((new_w_hist1 / N) - (new_w_hist0 / N)).abs().sum().item()
        assert new_l1distw < 1e-6, (
            f"Master histogram should NOT change (same fine bin), got {new_l1distw}"
        )

        # Degeneracy is broken: ratio >> 1, oscillation detected.
        new_ratio = l1distqw / (new_l1distw + 1e-12)
        assert new_ratio > 100.0, (
            f"New 29-bin ratio should be >> 1, got {new_ratio}"
        )

    def test_multibin_baseline_below_one(self):
        """With realistic multi-bin distributions, the baseline ratio is < 1.

        Because the denominator (29-bin master) has finer resolution than
        the numerator (15-bin qw), the same weight shift produces a larger
        L1 in master space → ratio < 1 at baseline.  This is expected and
        by design: oscillation is detected as ratio spiking ABOVE this
        baseline (as confirmed by Anjulie/Jinhang in kitchen MR !476).
        """
        N = 1024
        dev = torch.device(_DEVICE)
        torch.manual_seed(42)

        # Step 0: weights spread across many bins.
        latent0 = torch.randn(N, dtype=torch.float32, device=_DEVICE) * 2.0

        # Step 1: small perturbation (normal weight evolution, not oscillation).
        latent1 = latent0 + torch.randn(N, dtype=torch.float32, device=_DEVICE) * 0.1

        # qw histograms (15-bin FP4)
        qw_hist0 = _fp4_bin_histogram(_latent_to_fp4_bin_index(latent0, dev)).float()
        qw_hist1 = _fp4_bin_histogram(_latent_to_fp4_bin_index(latent1, dev)).float()
        l1distqw = ((qw_hist1 / N) - (qw_hist0 / N)).abs().sum().item()

        # w histograms (29-bin fine)
        w_hist0 = _master_w_bin_histogram(_latent_to_master_bin_index(latent0, dev)).float()
        w_hist1 = _master_w_bin_histogram(_latent_to_master_bin_index(latent1, dev)).float()
        l1distw = ((w_hist1 / N) - (w_hist0 / N)).abs().sum().item()

        # With finer bins, master L1 >= qw L1 → ratio ≤ 1 at baseline.
        assert l1distw > 0, "Master histogram should change with weight evolution"
        assert l1distqw > 0, "Quant histogram should also change"
        ratio = l1distqw / (l1distw + 1e-12)
        assert ratio < 1.0, (
            f"Baseline ratio with multi-bin distributions should be < 1, got {ratio}"
        )
        # But not zero — it should be a meaningful fraction.
        assert ratio > 0.1, f"Baseline ratio should not be negligible, got {ratio}"

    def test_multibin_boundary_oscillation_spikes_above_baseline(self):
        """Boundary oscillation produces ratio well above the ~0.5 baseline.

        Even with multi-bin distributions, concentrating weight near an FP4
        boundary and flipping the quantized assignment produces a clear
        ratio spike above the normal-evolution baseline.
        """
        N = 1024
        dev = torch.device(_DEVICE)
        torch.manual_seed(42)

        # Baseline: normal weight evolution (same as above).
        latent0 = torch.randn(N, dtype=torch.float32, device=_DEVICE) * 2.0
        latent1_normal = latent0 + torch.randn(N, dtype=torch.float32, device=_DEVICE) * 0.1
        qw_n0 = _fp4_bin_histogram(_latent_to_fp4_bin_index(latent0, dev)).float() / N
        qw_n1 = _fp4_bin_histogram(_latent_to_fp4_bin_index(latent1_normal, dev)).float() / N
        w_n0 = _master_w_bin_histogram(_latent_to_master_bin_index(latent0, dev)).float() / N
        w_n1 = _master_w_bin_histogram(_latent_to_master_bin_index(latent1_normal, dev)).float() / N
        baseline_ratio = (qw_n1 - qw_n0).abs().sum().item() / (
            (w_n1 - w_n0).abs().sum().item() + 1e-12
        )

        # Oscillation scenario: all weights near boundary 0.75, flipping.
        latent0_osci = torch.full((N,), 0.74, dtype=torch.float32, device=_DEVICE)
        latent1_osci = torch.full((N,), 0.76, dtype=torch.float32, device=_DEVICE)
        qw_o0 = _fp4_bin_histogram(_latent_to_fp4_bin_index(latent0_osci, dev)).float() / N
        qw_o1 = _fp4_bin_histogram(_latent_to_fp4_bin_index(latent1_osci, dev)).float() / N
        w_o0 = _master_w_bin_histogram(_latent_to_master_bin_index(latent0_osci, dev)).float() / N
        w_o1 = _master_w_bin_histogram(_latent_to_master_bin_index(latent1_osci, dev)).float() / N
        osci_ratio = (qw_o1 - qw_o0).abs().sum().item() / (
            (w_o1 - w_o0).abs().sum().item() + 1e-12
        )

        # Oscillation ratio should be dramatically higher than baseline.
        assert osci_ratio > baseline_ratio * 10, (
            f"Oscillation ratio ({osci_ratio:.2f}) should be >> baseline ({baseline_ratio:.2f})"
        )

    def test_stat_names_constant(self):
        assert OSCILLATION_STAT_NAMES == {
            "oscillation_ratio",
            "oscillation_ratio_reduced",
            "oscillation_l1distqw",
            "oscillation_l1distw",
            "oscillation_l1distw_reduced",
        }
