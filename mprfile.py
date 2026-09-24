"""
mprfile — read Refeyn mass-photometry .mpr files (AcquireMP / DiscoverMP) in Python.

Format notes (reverse-engineered from AcquireMP 2025.1.2 files, format_version 4):
  * A .mpr file is an ordinary HDF5 file with three top-level groups:
      movie/     raw acquisition + instrument configuration + sample info
      analysis/  AcquireMP's automatic particle analysis (fitted events, frame scores)
      display/   histogram / Gaussian-fit state of the GUI
  * movie/frame is (n_frames, ny, nx) uint16, compressed with the Zstandard HDF5
    filter (id 32015) -> needs `hdf5plugin`.
  * movie/frame carries attribute encoding='TIME_DIFFERENCE': frame 0 is stored
    as-is, every later frame is stored as (frame[i] - frame[i-1]) mod 2**16.
    Decoding = cumulative sum in uint16 (wrap-around arithmetic).
    movie/keyframe holds fully decoded frames every 100 frames (indices in its
    `indices` attribute), which allows random access and is used for verification.
  * Timestamps (movie/timeStack) are in nanoseconds; each stored frame is the sum
    of `frame_binning` camera exposures, so timeStack has one column per exposure.
  * Python None values are written as `<name>__is_none__` datasets and
    lists/tuples as groups containing an `__is_list__` marker with children
    '0', '1', ...; `metadata()` converts these back.

Dependencies: numpy, h5py, hdf5plugin   (pandas optional, for events_df()).

Quick start:
    from mprfile import MPRFile
    with MPRFile("002_Ladder.mpr") as m:
        print(m.summary())
        ev = m.events()               # dict of numpy arrays (contrast, x, y, frame ...)
        movie = m.frames()            # decoded raw movie, (n, ny, nx) uint16
        r = m.ratiometric()           # ratiometric contrast movie, float32, same indexing as events
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import h5py

try:  # registers the Zstandard filter with HDF5
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover
    hdf5plugin = None

__all__ = ["MPRFile", "Calibration", "load_events"]


# --------------------------------------------------------------------------- helpers
def _decode_scalar(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    return v


def _read_node(node: h5py.Group | h5py.Dataset) -> Any:
    """Recursively convert an HDF5 node to Python objects, undoing the
    __is_none__ / __is_list__ conventions AcquireMP uses."""
    if isinstance(node, h5py.Dataset):
        return _decode_scalar(node[()]) if node.shape == () else node[()]

    keys = list(node.keys())
    if "__is_list__" in keys:
        items = {}
        for k in keys:
            if k == "__is_list__":
                continue
            if k.endswith("__is_none__"):
                items[int(k[: -len("__is_none__")])] = None
            else:
                items[int(k)] = _read_node(node[k])
        return [items[i] for i in sorted(items)]

    out: dict[str, Any] = {}
    for k in keys:
        if k.endswith("__is_none__"):
            out[k[: -len("__is_none__")]] = None
        else:
            out[k] = _read_node(node[k])
    return out


# --------------------------------------------------------------------------- calibration
class Calibration:
    """Linear contrast -> mass calibration:  mass = slope * contrast + intercept."""

    def __init__(self, slope: float, intercept: float = 0.0, unit: str = "kDa"):
        self.slope, self.intercept, self.unit = float(slope), float(intercept), unit

    @classmethod
    def fit(cls, contrasts, masses, unit: str = "kDa") -> "Calibration":
        """Least-squares fit from known (contrast, mass) pairs, e.g. peak
        positions from `gaussian_fits()` and the known masses of your standard."""
        c, m = np.asarray(contrasts, float), np.asarray(masses, float)
        if len(c) < 2:
            raise ValueError("need at least two calibration points")
        slope, intercept = np.polyfit(c, m, 1)
        cal = cls(slope, intercept, unit)
        cal.r2 = 1 - np.sum((m - cal(c)) ** 2) / np.sum((m - m.mean()) ** 2)
        return cal

    def __call__(self, contrast):
        return self.slope * np.asarray(contrast, float) + self.intercept

    def inverse(self, mass):
        return (np.asarray(mass, float) - self.intercept) / self.slope

    def __repr__(self):
        r2 = f", R²={self.r2:.5f}" if hasattr(self, "r2") else ""
        return f"Calibration(mass = {self.slope:.6g}·contrast + {self.intercept:.6g} {self.unit}{r2})"


# --------------------------------------------------------------------------- main class
class MPRFile:
    """Read-only access to a Refeyn .mpr file."""

    def __init__(self, path: str | os.PathLike):
        if hdf5plugin is None:
            raise ImportError("hdf5plugin is required for the Zstandard-compressed movie: "
                              "pip install hdf5plugin")
        self.path = os.fspath(path)
        self.h5 = h5py.File(self.path, "r")
        self._movie_cache: np.ndarray | None = None

    # context manager ------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._movie_cache = None
        if self.h5.id.valid:
            self.h5.close()

    def __repr__(self):
        return f"MPRFile({os.path.basename(self.path)!r}, {self.n_frames} frames, {self.frame_shape})"

    # metadata -------------------------------------------------------------
    def metadata(self, group: str = "/") -> Any:
        """Full metadata tree (or a sub-tree, e.g. 'movie/configuration') as nested
        dicts/lists. Large arrays (movie, events, scores) are included as numpy
        arrays only if you ask for their group explicitly."""
        if group in ("/", ""):
            skip = {"movie/frame", "movie/keyframe", "movie/timeStack", "movie/systemTimeStack",
                    "movie/frameId", "movie/autofocus", "movie/autofocus_image",
                    "analysis/events", "analysis/scores"}

            def walk(g, prefix=""):
                if "__is_list__" in g:
                    return _read_node(g)
                d = {}
                for k in g:
                    full = f"{prefix}{k}"
                    if full in skip:
                        continue
                    if k.endswith("__is_none__"):
                        d[k[: -len("__is_none__")]] = None
                    elif isinstance(g[k], h5py.Group):
                        d[k] = walk(g[k], full + "/")
                    else:
                        d[k] = _read_node(g[k])
                return d

            return walk(self.h5)
        return _read_node(self.h5[group])

    @property
    def sample_info(self) -> dict:
        info = self.metadata("movie/measurement_info")
        info["comments"] = _decode_scalar(self.h5["movie/comments"][()])
        info["timestamp"] = _decode_scalar(self.h5["movie/timestamp"][()])
        return info

    @property
    def camera(self) -> dict:
        return self.metadata("movie/configuration/acq_camera")

    @property
    def instrument(self) -> dict:
        return self.metadata("movie/device_serials")

    @property
    def analysis_params(self) -> dict:
        return self.metadata("analysis/analysis_params")

    def summary(self) -> str:
        s, cam, ins = self.sample_info, self.camera, self.instrument
        ev = self.h5["analysis/events/events_fitted/contrasts"].shape[0] \
            if "analysis/events/events_fitted" in self.h5 else 0
        t = self.times()
        return "\n".join([
            f"File        : {os.path.basename(self.path)}",
            f"Instrument  : {ins.get('InstrumentName')}  (software {self.h5['movie/app_info/version'][()].decode()})",
            f"Recorded    : {s.get('timestamp')}  by {s.get('operatorName')}",
            f"Sample      : {s.get('sample')}  in {s.get('buffer')} pH {s.get('bufferpH')}",
            f"Comments    : {s.get('comments')}",
            f"Movie       : {self.n_frames} frames of {self.frame_shape[1]}x{self.frame_shape[0]} px, "
            f"{self.frame_rate:.2f} fps effective ({cam['frame_rate']:.1f} Hz camera, "
            f"binning {cam['frame_binning']} frames / {cam['pixel_binning']} px), {t[-1]:.1f} s",
            f"Events      : {ev} fitted events",
        ])

    # movie ----------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self.h5["movie/frame"].shape[0]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return tuple(self.h5["movie/frame"].shape[1:])

    @property
    def frame_rate(self) -> float:
        """Effective rate of the stored (binned) frames, in Hz."""
        cam = self.camera
        return cam["frame_rate"] / cam["frame_binning"]

    def times(self, relative: bool = True) -> np.ndarray:
        """Time of each stored frame in seconds (mean of its sub-exposures)."""
        ts = self.h5["movie/timeStack"][:].mean(axis=1) * 1e-9
        return ts - ts[0] if relative else ts

    def _encoding(self) -> str:
        enc = self.h5["movie/frame"].attrs.get("encoding", b"")
        return enc.decode() if isinstance(enc, bytes) else str(enc)

    def frames(self, start: int = 0, stop: int | None = None, cache: bool = True) -> np.ndarray:
        """Decoded raw movie frames[start:stop] as uint16 (n, ny, nx).
        The full movie is ~n*ny*nx*2 bytes (≈50 MB for a typical 1-minute file)."""
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        if self._movie_cache is not None:
            return self._movie_cache[start:stop]
        enc = self._encoding()
        if enc not in ("TIME_DIFFERENCE", ""):
            raise NotImplementedError(f"unknown frame encoding {enc!r}")
        if enc == "":
            return self.h5["movie/frame"][start:stop]
        if cache and start == 0 and stop == self.n_frames:
            movie = np.cumsum(self.h5["movie/frame"][:], axis=0, dtype=np.uint16)
            self._movie_cache = movie
            return movie
        # random access: start from the nearest keyframe at or before `start`
        k0 = self._keyframe_before(start)
        kf_idx, kf_pos = k0
        base = self.h5["movie/keyframe"][kf_pos] if kf_pos is not None else np.zeros(self.frame_shape, np.uint16)
        first = kf_idx + 1 if kf_pos is not None else 0
        diffs = self.h5["movie/frame"][first:stop]
        out = np.cumsum(np.concatenate([base[None], diffs]) if kf_pos is not None else diffs,
                        axis=0, dtype=np.uint16)
        offset = start - (kf_idx if kf_pos is not None else 0)
        return out[offset:]

    def frame(self, i: int) -> np.ndarray:
        """Single decoded frame (random access via keyframes)."""
        if i < 0:
            i += self.n_frames
        return self.frames(i, i + 1, cache=False)[0]

    def _keyframe_before(self, i: int):
        if "movie/keyframe" not in self.h5:
            return 0, None
        idx = np.asarray(self.h5["movie/keyframe"].attrs["indices"])
        valid = np.nonzero(idx <= i)[0]
        if len(valid) == 0:
            return 0, None
        j = valid[-1]
        return int(idx[j]), int(j)

    def verify(self) -> bool:
        """Check the decoded movie against every stored keyframe."""
        movie = self.frames()
        idx = self.h5["movie/keyframe"].attrs["indices"]
        kf = self.h5["movie/keyframe"]
        return all(np.array_equal(movie[i], kf[j]) for j, i in enumerate(idx) if i < len(movie))

    def ratiometric(self, n_avg: int | None = None, acquiremp_sign: bool = True) -> np.ndarray:
        """Sliding ratiometric contrast movie, same shape as the raw movie:
            r[i] = mean(F[i : i+n]) / mean(F[i-n : i]) - 1
        so r[i] shows what changed at raw frame i — the same frame index AcquireMP
        stores in events()['frame']. The first n and last n-1 frames are NaN.

        With acquiremp_sign=True (default) the sign follows AcquireMP's convention
        (binding = negative contrast, unbinding = positive), which depends on the
        instrument's `scatter_interference_destructive` setting. n_avg defaults to
        the value AcquireMP used for its analysis."""
        n = int(n_avg or self.analysis_params.get("n_avg", 5))
        movie = self.frames()
        N = movie.shape[0]
        if N < 2 * n:
            raise ValueError("too few frames for this n_avg")
        cs = np.zeros((N + 1,) + movie.shape[1:], np.float64)
        np.cumsum(movie, axis=0, dtype=np.float64, out=cs[1:])
        i = np.arange(n, N - n + 1)
        before = cs[i] - cs[i - n]
        after = cs[i + n] - cs[i]
        out = np.full(movie.shape, np.nan, np.float32)
        out[n:N - n + 1] = after / before - 1
        if acquiremp_sign and not bool(self.h5["movie/configuration/scatter_interference_destructive"][()]):
            np.negative(out, out=out)
        return out

    def autofocus_image(self) -> np.ndarray:
        return self.h5["movie/autofocus_image"][:]

    # analysis -------------------------------------------------------------
    def events(self, only_good: bool = True, calibration: Calibration | None = None) -> dict[str, np.ndarray]:
        """Fitted landing/unbinding events from AcquireMP's analysis.
        Keys: contrast, x, y (px), frame, time (s), fit_error, residual_error,
        nn_distance (px), id, good, selected (+ mass if a calibration is given
        or stored in the file)."""
        g = self.h5["analysis/events/events_fitted"]
        ev = {
            "contrast": g["contrasts"][:],
            "x": g["x_coords"][:],
            "y": g["y_coords"][:],
            "frame": g["frame_indices"][:],
            "fit_error": g["fit_errors"][:],
            "residual_error": g["residual_errors"][:],
            "nn_distance": g["nearest_neighbour_distances"][:],
            "id": g["ids"][:],
            "good": g["good_flags"][:].astype(bool),
            "selected": g["selections"][:].astype(bool),
        }
        t = self.times()
        ev["time"] = np.interp(ev["frame"], np.arange(len(t)), t)
        stored = g["calibrated_values"][:]
        if calibration is not None:
            ev["mass"] = calibration(ev["contrast"])
        elif np.isfinite(stored).any():
            ev["mass"] = stored
        if only_good:
            keep = ev["good"] & ev["selected"]
            ev = {k: v[keep] for k, v in ev.items()}
        return ev

    def events_df(self, **kw):
        """events() as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.events(**kw))

    def scores(self) -> dict[str, np.ndarray]:
        """Per-frame quality scores (brightness, sharpness, saturation, ...)."""
        return {k: v[:] for k, v in self.h5["analysis/scores"].items()}

    def gaussian_fits(self) -> list[dict]:
        """Gaussian peaks fitted in the GUI histogram, sorted by position (contrast
        units unless the histogram was in mass mode, see display/histogram/xAxisMode)."""
        base = "display/histogram/overlaysState/GAUSSIAN_FIT"
        if base not in self.h5:
            return []
        out = []
        for k, grp in self.h5[base].items():
            gp = grp["gaussianPeak"]
            out.append({
                "id": int(k),
                "position": float(gp["pos"][()]),
                "sigma": float(gp["sigma"][()]),
                "counts": float(gp["counts"][()]),
                "range": (float(grp["cMinSelection"][()]), float(grp["cMaxSelection"][()])),
            })
        return sorted(out, key=lambda d: d["position"])

    def histogram(self, bin_width: float = 2e-4, key: str = "contrast", events: dict | None = None,
                  range: tuple[float, float] | None = None):
        """Histogram of event contrasts (or masses with key='mass').
        Returns (counts, bin_edges)."""
        ev = events if events is not None else self.events()
        v = ev[key]
        lo, hi = range if range else (np.nanmin(v), np.nanmax(v))
        edges = np.arange(lo, hi + bin_width, bin_width)
        return np.histogram(v, bins=edges)

    # export ---------------------------------------------------------------
    def export_events_csv(self, path: str, **kw) -> str:
        ev = self.events(**kw)
        cols = list(ev)
        data = np.column_stack([ev[c].astype(float) for c in cols])
        np.savetxt(path, data, delimiter=",", header=",".join(cols), comments="", fmt="%.10g")
        return path

    def export_movie_tiff(self, path: str, ratiometric: bool = False, **kw) -> str:
        """Write the raw (uint16) or ratiometric (float32) movie as a multi-page TIFF
        for ImageJ/Fiji. Requires `tifffile`."""
        import tifffile
        arr = self.ratiometric(**kw) if ratiometric else self.frames()
        tifffile.imwrite(path, arr, imagej=True, metadata={"axes": "TYX", "finterval": 1 / self.frame_rate})
        return path


def load_events(path: str, **kw) -> dict[str, np.ndarray]:
    """One-liner: events of a .mpr file."""
    with MPRFile(path) as m:
        return m.events(**kw)
