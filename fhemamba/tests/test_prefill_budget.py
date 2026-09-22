"""Structural sanity of the scan-prefill budget (numerics proven elsewhere)."""

import pytest
from fhemamba.prefill_budget import Dims, compare, prefill_budget


def test_scan_prefill_wins_on_depth_and_bootstraps() -> None:
    r = compare(seq_len=128)
    seq, pre = r["sequential_decode_of_prompt"], r["scan_prefill"]
    assert pre["recurrence_depth_total"] == 8  # log2(64) + 2 SERIAL chunks
    assert seq["recurrence_depth_total"] == 128
    assert pre["bootstraps"] < seq["bootstraps"] / 3
    assert pre["ct_pt"] < seq["ct_pt"] / 4  # time-batched matmuls dominate the win


def test_scan_recurrence_mults_cost_more_but_bounded() -> None:
    # Known trade: Hillis-Steele B-lineage pays ~log2(L) big mults per tile.
    r64 = compare(seq_len=64)
    r512 = compare(seq_len=512)
    for r in (r64, r512):
        assert r["scan_prefill"]["ct_ct"] < r["sequential_decode_of_prompt"]["ct_ct"] * 2
    # Bootstraps stay ~linear in T, but the per-token constant beats decode's
    per_token_prefill = r512["scan_prefill"]["bootstraps"] / 512
    per_token_decode = r512["sequential_decode_of_prompt"]["bootstraps"] / 512
    assert per_token_prefill < per_token_decode / 3


def test_memoryless_heads_reduce_state_work() -> None:
    with_kill = compare(seq_len=64, killed_head_fraction=49 / 576)
    without = compare(seq_len=64, killed_head_fraction=0.0)
    assert with_kill["scan_prefill"]["ct_ct"] <= without["scan_prefill"]["ct_ct"]


def test_depth_accounts_for_serial_carry_and_partial_chunks() -> None:
    for length, depth, chunks in ((1, 1, 1), (2, 2, 1), (64, 7, 1), (65, 8, 2), (512, 14, 8)):
        budget = prefill_budget(Dims(), {}, length)
        assert budget["recurrence_depth_per_layer"] == depth
        assert budget["chunks"] == chunks
        assert budget["state_tiles"] == 6 * length
        assert budget["native_execution_measured"] is False
    assert compare(512)["scan_prefill"]["assumes_compacted_heads"] is False


@pytest.mark.parametrize(
    "kwargs", [{"seq_len": 0}, {"chunk": 0}, {"time_batch": 9}, {"killed_head_fraction": -0.1}]
)
def test_invalid_prefill_geometry_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"positive|slot layout|in "):
        prefill_budget(Dims(), {}, **({"seq_len": 64} | kwargs))
