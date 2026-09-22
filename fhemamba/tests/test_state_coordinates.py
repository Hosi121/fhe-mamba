import numpy as np
import pytest
from fhemamba.state_coordinates import regularize_row_scales


def test_regularization_bounds_amplification_without_losing_observed_coverage():
    rows = np.array([[0, 1, 0.01], [8, 0.02, 0.1], [64, 0, 4], [0, 3, 2]])
    scales = regularize_row_scales(rows, 2, 0.125)
    assert (scales >= rows).all()
    groups = rows.reshape(2, -1).max(axis=1, keepdims=True)
    assert (groups / scales.reshape(2, -1) <= 8).all()
    assert np.array_equal(scales.reshape(2, -1).max(axis=1), groups[:, 0])
    # The rule is homogeneous away from the absolute zero floor.
    assert np.allclose(regularize_row_scales(2 * rows, 2, 0.125), 2 * scales)
    assert np.all(regularize_row_scales(rows, 2, 1).reshape(2, -1) == groups)


def test_regularized_coordinates_preserve_update_and_readout():
    rng = np.random.default_rng(37)
    scale = regularize_row_scales(np.abs(rng.normal(size=(4, 3))), 2, 0.125)
    state = rng.normal(size=(4, 3, 7))
    decay = rng.uniform(size=4)
    write = rng.normal(size=(4, 3, 7))
    read = rng.normal(size=(4, 7))
    expected = np.einsum("hpn,hn->hp", decay[:, None, None] * state + write, read)
    normalized = decay[:, None, None] * (state / scale[..., None]) + write / scale[..., None]
    actual = scale * np.einsum("hpn,hn->hp", normalized, read)
    assert np.allclose(actual, expected, atol=1e-14)
    assert np.isfinite(regularize_row_scales(np.zeros((4, 3)), 2)).all()
    assert (regularize_row_scales(np.zeros((4, 3)), 2) > 0).all()


@pytest.mark.parametrize(
    ("rows", "heads", "floor"),
    [
        ([[1, -1]], 1, 0.1),
        ([[1, np.nan]], 1, 0.1),
        ([[1, np.inf]], 1, 0.1),
        ([[1, 2]], 2, 0.1),
        ([[1, 2]], 0, 0.1),
        ([[1, 2]], 1, -0.1),
        ([[1, 2]], 1, 1.1),
        ([[1, 2]], 1, np.nan),
        ([], 1, 0.1),
    ],
)
def test_invalid_coordinate_scales_fail_closed(rows, heads, floor):
    with pytest.raises(ValueError, match="finite nonnegative"):
        regularize_row_scales(np.asarray(rows), heads, floor)
