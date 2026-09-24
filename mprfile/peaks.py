"""
mprfile.peaks — histogram peak detection and Gaussian-mixture fitting.

Works on any 1-D values (contrasts or masses). Peaks are found on a lightly smoothed
histogram and then all fitted at once as a sum of Gaussians (optionally on top of one
broad background Gaussian) with scipy's bounded least squares.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

__all__ = ["gauss", "multi_gauss", "PeakFit", "fit_peaks"]


def gauss(x, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def multi_gauss(x, *p):
    out = np.zeros_like(np.asarray(x, float))
    for i in range(0, len(p), 3):
        out += gauss(x, *p[i:i + 3])
    return out


@dataclass
class PeakFit:
    """Result of fit_peaks().

    peaks: one row per peak — position, position_err, sigma, counts (events under
           the Gaussian), fraction (counts / all values in the fitted range).
    edges, counts: the histogram that was fitted.
    params: flat Gaussian parameters [amplitude, mu, sigma, ...] (background first
            if background=True), usable with multi_gauss(x, *params).
    """
    peaks: pd.DataFrame
    edges: np.ndarray
    counts: np.ndarray
    params: np.ndarray
    background: np.ndarray | None = None
    n_total: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def centers(self):
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def bin_width(self):
        return float(self.edges[1] - self.edges[0])

    def curve(self, x):
        """Total fitted curve (all peaks + background), in counts per bin."""
        return multi_gauss(x, *self.params)

    def peak_curves(self, x):
        """List of individual peak curves (without background)."""
        k = 3 if self.background is not None else 0
        return [gauss(x, *self.params[i:i + 3]) for i in range(k, len(self.params), 3)]


def fit_peaks(values, lo: float, hi: float, bin_width: float, *,
              smooth_bins: float = 2.0, min_prominence: float = 0.04,
              min_counts: int = 20, min_fraction: float = 0.02,
              sigma_guess: float | None = None, sigma_max: float | None = None,
              max_shift_bins: float = 4.0, background: bool = False,
              max_peaks: int = 12, detect_pad_bins: int = 10) -> PeakFit:
    """Find and fit Gaussian peaks in `values` within [lo, hi].

    min_prominence: peak must stand out by this fraction of the tallest (smoothed) bin.
    min_counts / min_fraction: peaks with fewer events, or a smaller share of all
        values in range, are dropped after fitting.
    sigma_guess / sigma_max: starting width and upper limit (default 3 and 15 bins).
    max_shift_bins: how far a fitted centre may move from the detected maximum.
    background: add one broad Gaussian that absorbs unresolved events between peaks.
    detect_pad_bins: peak *detection* also looks this many bins below `lo`, so a peak just
        inside the lower limit is still recognised (the fit itself stays within [lo, hi]).
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    edges = np.arange(lo, hi + bin_width, bin_width)
    counts, _ = np.histogram(v, bins=edges)
    x = 0.5 * (edges[1:] + edges[:-1])
    n_total = int(counts.sum())
    empty = pd.DataFrame(columns=["position", "position_err", "sigma", "counts", "fraction"])
    if n_total == 0:
        return PeakFit(empty, edges, counts, np.array([]), None, 0, ["no events in range"])

    smooth = gaussian_filter1d(counts.astype(float), smooth_bins) if smooth_bins else counts.astype(float)
    # detection on a histogram padded below `lo` (not below 0 for positive data)
    pad = int(min(detect_pad_bins, max(lo, 0) // bin_width)) if lo >= 0 else detect_pad_bins
    d_edges = np.arange(lo - pad * bin_width, hi + bin_width / 2, bin_width)
    d_counts, _ = np.histogram(v, bins=d_edges)
    d_smooth = gaussian_filter1d(d_counts.astype(float), smooth_bins) if smooth_bins else d_counts.astype(float)
    d_idx, props = find_peaks(d_smooth, prominence=max(min_prominence * smooth.max(), 1.0))
    keep_d = d_idx >= pad                                   # peak maximum inside [lo, hi]
    d_idx, props = d_idx[keep_d], {k: v_[keep_d] for k, v_ in props.items()}
    idx = np.clip(d_idx - pad, 0, len(smooth) - 1)
    if len(idx) > max_peaks:
        idx = idx[np.argsort(props["prominences"])[::-1][:max_peaks]]
        idx.sort()
    if len(idx) == 0:
        return PeakFit(empty, edges, counts, np.array([]), None, n_total, ["no peaks found"])

    sg = sigma_guess or 3 * bin_width
    smax = sigma_max or 15 * bin_width
    notes: list[str] = []

    def _fit(centres, heights):
        p0, lower, upper = [], [], []
        if background:
            p0 += [max(np.percentile(smooth, 20), 0.5), float(np.mean(centres)), (hi - lo) / 4]
            lower += [0, lo, (hi - lo) / 20]
            upper += [np.inf, hi, hi - lo]
        for c, h in zip(centres, heights):
            p0 += [max(h, 1e-3), c, min(sg, smax * 0.99)]
            lower += [0, c - max_shift_bins * bin_width, bin_width / 2]
            upper += [np.inf, c + max_shift_bins * bin_width, smax]
        try:
            popt, pcov = curve_fit(multi_gauss, x, counts, p0=p0, bounds=(lower, upper), maxfev=50000)
            perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
        except RuntimeError as exc:  # no convergence: fall back to the start values
            popt, perr = np.array(p0), np.full(len(p0), np.nan)
            notes.append(f"Gaussian fit did not converge ({exc}); showing start values")
        k = 3 if background else 0
        pk, pe = popt[k:].reshape(-1, 3), perr[k:].reshape(-1, 3)
        n_ev = pk[:, 0] * pk[:, 2] * np.sqrt(2 * np.pi) / bin_width
        df = pd.DataFrame({"position": pk[:, 1], "position_err": pe[:, 1], "sigma": pk[:, 2],
                           "counts": n_ev, "fraction": n_ev / n_total})
        return popt, df

    centres, heights = x[idx], smooth[idx]
    popt, df = _fit(centres, heights)
    keep = ((df.counts >= min_counts) & (df.fraction >= min_fraction)).values
    if not keep.all():
        notes.append(f"{int((~keep).sum())} minor peak(s) below min_counts={min_counts} / "
                     f"min_fraction={min_fraction:g} left out")
        if keep.any():
            popt, df = _fit(df.position.values[keep], (df.counts.values[keep] * bin_width
                            / (df.sigma.values[keep] * np.sqrt(2 * np.pi))))
        else:
            return PeakFit(empty, edges, counts, np.array([]), None, n_total, notes)
    order = np.argsort(df.position.values)
    k = 3 if background else 0
    pk = popt[k:].reshape(-1, 3)[order]
    params = np.concatenate([popt[:3], pk.ravel()]) if background else pk.ravel()
    df = df.iloc[order].reset_index(drop=True)
    return PeakFit(df, edges, counts, params, popt[:3] if background else None, n_total, notes)
