"""
Compute normalization statistics (mean/std/percentiles) for datasets that conform to DCA Specs.
"""

import logging

import numpy as np
from dask import compute as dask_compute

# Percentiles to compute
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]
SPATIAL_DIMS = ["Z", "Y", "X"]


logger = logging.getLogger(__name__)


def _make_stats_dict(mean, std, percentiles) -> dict:
    """Build statistics dictionary from computed values."""
    p1, p5, p25, p50, p75, p95, p99 = percentiles
    return {
        "mean": float(mean),
        "std": float(std),
        "median": float(p50),
        "iqr": float(p75 - p25),
        "p5": float(p5),
        "p95": float(p95),
        "p95_p5": float(p95 - p5),
        "p1": float(p1),
        "p99": float(p99),
        "p99_p1": float(p99 - p1),
    }


def compute_stats_numpy(array: np.ndarray, target_dims: list[str]) -> dict:
    """
    Fast in-memory computation using pure numpy. Does not do any subsampling
    """
    dim_map = {dim: i for i, dim in enumerate(target_dims)}
    c_axis, t_axis = dim_map["C"], dim_map["T"]  # Assume C and T are present

    if t_axis != 0:
        raise ValueError(
            "Expected time dimension 'T' to be the first dimension (axis 0)"
        )

    n_channels = array.shape[c_axis]
    n_timepoints = array.shape[t_axis]

    all_stats = {}

    for ch in range(n_channels):
        logger.info(f"Channel {ch}")

        # Select channel
        ch_data = np.take(array, ch, axis=c_axis)

        # Dataset stats (all dims)
        ds_stats = _make_stats_dict(
            np.nanmean(ch_data),
            np.nanstd(ch_data),
            np.nanpercentile(ch_data, PERCENTILES),
        )
        # Per-timepoint stats (vectorized)
        tp_data = ch_data.reshape(n_timepoints, -1)

        tp_means = np.nanmean(tp_data, axis=1)
        tp_stds = np.nanstd(tp_data, axis=1)
        tp_percentiles = np.nanpercentile(tp_data, PERCENTILES, axis=1)

        tp_stats = {
            str(t): _make_stats_dict(tp_means[t], tp_stds[t], tp_percentiles[:, t])
            for t in range(n_timepoints)
        }

        all_stats[str(ch)] = {
            "dataset_statistics": ds_stats,
            "timepoint_statistics": tp_stats,
        }

    return all_stats


def compute_stats_dask(da_xr, target_dims: list[str], grid: int) -> dict:
    """Out-of-memory computation using dask/xarray."""

    n_channels = da_xr.sizes["C"]  # assume that C and T are present
    n_timepoints = da_xr.sizes["T"]

    # Subsampling indexers
    subsample = {d: slice(0, None, grid) for d in SPATIAL_DIMS if d in da_xr.dims}

    all_stats = {}

    for ch in range(n_channels):
        # Select channel
        ch_da = da_xr.isel(C=ch)
        ch_sub = ch_da.isel(subsample) if subsample else ch_da

        # Dataset stats
        qs = np.array(PERCENTILES) / 100.0
        quantiles_da = ch_sub.quantile(qs, skipna=True)
        mean_da = ch_sub.mean(skipna=True)
        std_da = ch_sub.std(skipna=True)
        quantiles, mean_val, std_val = dask_compute(quantiles_da, mean_da, std_da)

        ds_stats = _make_stats_dict(mean_val.item(), std_val.item(), quantiles.values)

        # Per-timepoint stats
        tp_stats = {}
        reduce_dims = [d for d in ch_da.dims if d != "T"]

        for t in range(n_timepoints):
            tp_da = ch_da.isel(T=t)
            tp_sub = tp_da.isel(subsample)

            q_da = tp_sub.quantile(qs, dim=reduce_dims, skipna=True)
            m_da = tp_sub.mean(dim=reduce_dims, skipna=True)
            s_da = tp_sub.std(dim=reduce_dims, skipna=True)
            q_vals, m_val, s_val = dask_compute(q_da, m_da, s_da)

            tp_stats[str(t)] = _make_stats_dict(
                m_val.item(), s_val.item(), q_vals.values
            )

        all_stats[str(ch)] = {
            "dataset_statistics": ds_stats,
            "timepoint_statistics": tp_stats,
        }

    return all_stats
