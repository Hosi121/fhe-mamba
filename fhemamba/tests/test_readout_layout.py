from __future__ import annotations

import pytest
from fhemamba.readout_layout import (
    readout_output_slots,
    readout_reduce_mask,
    readout_reduce_steps,
    readout_scatter_mask,
    readout_scatter_shifts,
    required_readout_rotations,
    state_slot,
    state_slots,
)


def test_rank_major_state_layout() -> None:
    assert state_slots(4, 3) == 12
    assert tuple(
        state_slot(d_state=4, rank_index=rank, state_index=state)
        for rank in range(3)
        for state in range(4)
    ) == tuple(range(12))


def test_readout_layout_matches_4x3_golden_vectors() -> None:
    assert readout_reduce_steps(4) == (1, 2)
    assert readout_reduce_mask(d_state=4, mimo_rank=3, step=1) == (
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
    )
    assert readout_reduce_mask(d_state=4, mimo_rank=3, step=2) == (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    assert readout_scatter_mask(d_state=4, mimo_rank=3, rank_index=2) == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    assert readout_scatter_shifts(d_state=4, mimo_rank=3, dense_output=True) == (0, 3, 6)
    assert readout_output_slots(
        d_state=4,
        mimo_rank=3,
        readout_strategy="rank-local",
    ) == (0, 4, 8)
    assert required_readout_rotations(
        d_state=4,
        mimo_rank=3,
        readout_strategy="slotwise",
    ) == tuple(range(1, 10))
    assert required_readout_rotations(
        d_state=4,
        mimo_rank=3,
        readout_strategy="rank-reduce",
    ) == (1, 2, 3, 6)
    assert required_readout_rotations(
        d_state=4,
        mimo_rank=3,
        readout_strategy="rank-local",
    ) == (1, 2)


def test_non_power_of_two_state_layout_and_batch_padding() -> None:
    assert readout_reduce_steps(5) == (1, 2, 4)
    assert readout_scatter_shifts(d_state=5, mimo_rank=2, dense_output=True) == (0, 4)
    assert readout_output_slots(
        d_state=5,
        mimo_rank=2,
        readout_strategy="rank-local",
    ) == (0, 5)
    assert required_readout_rotations(
        d_state=5,
        mimo_rank=2,
        readout_strategy="rank-reduce",
    ) == (1, 2, 4)

    padded = readout_reduce_mask(d_state=5, mimo_rank=2, step=1, batch_size=16)
    assert len(padded) == 16
    assert padded[10:] == (0.0,) * 6


def test_layout_rejects_invalid_shapes_and_strategies() -> None:
    with pytest.raises(ValueError, match="d_state must be positive"):
        state_slots(0, 4)
    with pytest.raises(ValueError, match="mimo_rank must be positive"):
        state_slots(4, 0)
    with pytest.raises(ValueError, match="rank_index must be non-negative"):
        state_slot(d_state=4, rank_index=-1, state_index=0)
    with pytest.raises(ValueError, match="state_index"):
        state_slot(d_state=4, rank_index=0, state_index=4)
    with pytest.raises(ValueError, match="step must be positive"):
        readout_reduce_mask(d_state=4, mimo_rank=2, step=0)
    with pytest.raises(ValueError, match="batch_size"):
        readout_reduce_mask(d_state=4, mimo_rank=2, step=1, batch_size=4)
    with pytest.raises(ValueError, match="rank_index"):
        readout_scatter_mask(d_state=4, mimo_rank=2, rank_index=2)
    with pytest.raises(ValueError, match="unsupported readout_strategy"):
        required_readout_rotations(
            d_state=4,
            mimo_rank=2,
            readout_strategy="bad",  # type: ignore[arg-type]
        )
