"""
Unit tests for dca_helpers.normalization_stats.compute_stats_numpy.

NOTE: normalization_stats.py line 53 contains a `breakpoint()` that must be
removed before these tests can run.
"""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from dca_helpers.normalization_stats import compute_stats_dask, compute_stats_numpy

EXPECTED_STAT_KEYS = {
    "mean",
    "std",
    "median",
    "iqr",
    "p5",
    "p95",
    "p95_p5",
    "p1",
    "p99",
    "p99_p1",
}


@pytest.mark.unit
class TestComputeStatsNumpy:
    def test_output_structure_single_channel_single_timepoint(self):
        """
        Result should have channels -> dataset_statistics and timepoint_statistics.
        """
        array = np.ones((1, 1, 4, 4, 4), dtype=np.float32)
        result = compute_stats_numpy(array, ["T", "C", "Z", "Y", "X"])

        assert "0" in result

        ch = result["0"]
        assert "dataset_statistics" in ch
        assert "timepoint_statistics" in ch
        assert EXPECTED_STAT_KEYS == set(ch["dataset_statistics"].keys())
        assert "0" in ch["timepoint_statistics"]
        assert EXPECTED_STAT_KEYS == set(ch["timepoint_statistics"]["0"].keys())

    def test_uniform_array_mean_std_median(self):
        """Uniform array should give exact mean, std=0, and matching median."""
        array = np.full((1, 1, 4, 4, 4), 7.0, dtype=np.float32)
        result = compute_stats_numpy(array, ["T", "C", "Z", "Y", "X"])
        ds = result["0"]["dataset_statistics"]

        assert ds["mean"] == pytest.approx(7.0)
        assert ds["std"] == pytest.approx(0.0)
        assert ds["median"] == pytest.approx(7.0)
        assert ds["iqr"] == pytest.approx(0.0)
        assert ds["p95_p5"] == pytest.approx(0.0)
        assert ds["p99_p1"] == pytest.approx(0.0)

    def test_multiple_channels_computed_independently(self):
        """Each channel should reflect only its own data."""
        array = np.zeros((1, 2, 4, 4, 4), dtype=np.float32)
        array[:, 0, ...] = 1.0
        array[:, 1, ...] = 10.0
        result = compute_stats_numpy(array, ["T", "C", "Z", "Y", "X"])

        assert "0" in result
        assert "1" in result
        assert result["0"]["dataset_statistics"]["mean"] == pytest.approx(1.0)
        assert result["1"]["dataset_statistics"]["mean"] == pytest.approx(10.0)

    def test_multiple_timepoints_have_independent_stats(self):
        """Each timepoint entry should reflect only its own data."""
        array = np.zeros((3, 1, 4, 4, 4), dtype=np.float32)
        for t in range(3):
            array[t, 0, ...] = float(t)
        result = compute_stats_numpy(array, ["T", "C", "Z", "Y", "X"])

        tp_stats = result["0"]["timepoint_statistics"]
        assert set(tp_stats.keys()) == {"0", "1", "2"}
        assert tp_stats["0"]["mean"] == pytest.approx(0.0)
        assert tp_stats["1"]["mean"] == pytest.approx(1.0)
        assert tp_stats["2"]["mean"] == pytest.approx(2.0)


# ── _compute_stats_dask ────────────────────────────────────────────────────────


@pytest.fixture
def dask_xr_array():
    """2-timepoint, 1-channel, 8x8x8 dask-backed xarray DataArray."""

    rng = np.random.default_rng(42)
    data = rng.random((2, 1, 8, 8, 8)).astype(np.float32)
    return xr.DataArray(
        da.from_array(data, chunks=(1, 1, 8, 8, 8)),
        dims=["T", "C", "Z", "Y", "X"],
    )


@pytest.mark.integration
class TestComputeStatsDask:
    def test_stats_match_numpy_on_same_data(self, dask_xr_array):
        """Dask and numpy implementations should produce identical stats (grid=1)."""
        np_data = dask_xr_array.values  # eagerly load for numpy path
        numpy_result = compute_stats_numpy(np_data, ["T", "C", "Z", "Y", "X"])

        dask_result = compute_stats_dask(
            dask_xr_array, ["T", "C", "Z", "Y", "X"], grid=1
        )

        for ch in numpy_result:
            np_ds = numpy_result[ch]["dataset_statistics"]
            dk_ds = dask_result[ch]["dataset_statistics"]
            for key in np_ds:
                assert dk_ds[key] == pytest.approx(
                    np_ds[key], rel=1e-4
                ), f"dataset_statistics mismatch at [{ch}][{key!r}]"

            for t in numpy_result[ch]["timepoint_statistics"]:
                np_tp = numpy_result[ch]["timepoint_statistics"][t]
                dk_tp = dask_result[ch]["timepoint_statistics"][t]
                for key in np_tp:
                    assert dk_tp[key] == pytest.approx(
                        np_tp[key], rel=1e-4
                    ), f"timepoint stats mismatch at [{ch}][t={t}][{key!r}]"

    def test_multiple_channels_computed_independently(self):
        """Each channel should reflect only its own data."""
        # Modify dask_xr_array to have distinct values in each channel
        array = np.random.default_rng(42).random((1, 2, 4, 4, 4), dtype=np.float32)
        array[:, 0, ...] = 1.0
        array[:, 1, ...] = 10.0
        dask_xr_array = xr.DataArray(
            da.from_array(array, chunks=(1, 1, 8, 8, 8)),
            dims=["T", "C", "Z", "Y", "X"],
        )

        result = compute_stats_dask(dask_xr_array, ["T", "C", "Z", "Y", "X"], grid=1)

        assert "0" in result
        assert "1" in result
        assert result["0"]["dataset_statistics"]["mean"] == pytest.approx(1.0)
        assert result["1"]["dataset_statistics"]["mean"] == pytest.approx(10.0)

    def test_multiple_timepoints_have_independent_stats(self):
        """Each timepoint entry should reflect only its own data."""
        array = np.zeros((3, 1, 4, 4, 4), dtype=np.float32)
        for t in range(3):
            array[t, 0, ...] = float(t)

        dask_xr_array = xr.DataArray(
            da.from_array(array, chunks=(1, 1, 8, 8, 8)),
            dims=["T", "C", "Z", "Y", "X"],
        )

        result = compute_stats_dask(dask_xr_array, ["T", "C", "Z", "Y", "X"], grid=1)

        tp_stats = result["0"]["timepoint_statistics"]
        assert set(tp_stats.keys()) == {"0", "1", "2"}
        assert tp_stats["0"]["mean"] == pytest.approx(0.0)
        assert tp_stats["1"]["mean"] == pytest.approx(1.0)
        assert tp_stats["2"]["mean"] == pytest.approx(2.0)

    def test_grid_subsampling_reduces_data_used(self):
        """grid > 1 should only sample strided spatial positions."""
        array = np.zeros((3, 1, 4, 4, 4), dtype=np.float32)
        array[:, 0, ::2, ::2, ::2] = 10
        dask_xr_array = xr.DataArray(
            da.from_array(array, chunks=(1, 1, 8, 8, 8)),
            dims=["T", "C", "Z", "Y", "X"],
        )
        result = compute_stats_dask(dask_xr_array, ["T", "C", "Z", "Y", "X"], grid=2)
        assert result["0"]["dataset_statistics"]["mean"] == pytest.approx(10.0)
        tp_stats = result["0"]["timepoint_statistics"]
        assert tp_stats["0"]["mean"] == pytest.approx(10.0)

        result = compute_stats_dask(dask_xr_array, ["T", "C", "Z", "Y", "X"], grid=1)
        assert result["0"]["dataset_statistics"]["mean"] == pytest.approx(1.25)
        tp_stats = result["0"]["timepoint_statistics"]
        assert tp_stats["0"]["mean"] == pytest.approx(1.25)
