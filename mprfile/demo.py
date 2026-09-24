"""
mprfile.demo — realistic *simulated* .mpr files, for trying the package without data,
for the documentation and for tests.

    from mprfile.demo import make_demo_files
    ladder, sample = make_demo_files("demo_data")
    # -> demo_data/demo_ladder.mpr  (MassFerence-P1-like ladder: n × 86 kDa)
    #    demo_data/demo_sample.mpr  (demo protein: 150 / 300 / 450 kDa + low-mass population)

The files follow the AcquireMP layout (HDF5, Zstandard, TIME_DIFFERENCE-encoded movie,
fitted events, metadata), so every function in mprfile works on them. The movie is small
(a few MB) but shows real landing events: each event adds an interferometric spot at its
position from its frame onwards, on top of shot noise.
All numbers are made up; they only resemble typical TwoMP measurements.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

__all__ = ["make_demo_files", "write_demo_mpr", "DEMO_SLOPE"]

DEMO_SLOPE = -31000.0      # kDa per unit contrast (mass = slope * contrast + intercept)
DEMO_INTERCEPT = -2.5
FRAME_RATE, FRAME_BINNING = 475.0, 10


def _contrast(mass):
    return (np.asarray(mass, float) - DEMO_INTERCEPT) / DEMO_SLOPE


def _species(rng, specs, sigma_rel=0.035, sigma_min=10.0):
    """specs: list of (mass kDa, n events). Returns masses with instrument-like widths."""
    out = []
    for mass, n in specs:
        s = max(sigma_min, sigma_rel * mass)
        out.append(rng.normal(mass, s, int(n)))
    return np.concatenate(out)


def _psf(size=11, s1=1.4, s2=3.0):
    """Interferometric spot: bright core with a dark ring (raw frames, non-destructive
    interference). Normalised to a central value of 1."""
    y, x = np.mgrid[-size:size + 1, -size:size + 1]
    r2 = x ** 2 + y ** 2
    p = np.exp(-r2 / (2 * s1 ** 2)) - 0.3 * np.exp(-r2 / (2 * s2 ** 2))
    return p / p[size, size]


def write_demo_mpr(path, masses_bind, masses_unbind, *, sample: str, comments: str, seed: int = 0,
                   n_frames: int = 500, shape=(48, 120), is_calibrant: bool = False,
                   timestamp: str = "2026_01_01-12_00_00") -> str:
    # independent stream: must not reuse the seed that generated the masses (correlated noise)
    rng = np.random.default_rng([seed, 991])
    ny, nx = shape
    n_b, n_u = len(masses_bind), len(masses_unbind)
    n = n_b + n_u
    contrast = np.concatenate([_contrast(masses_bind), -_contrast(masses_unbind)])  # unbinding: positive
    contrast = contrast + rng.normal(0, 0.00012, n)                                  # extra fit scatter
    # landing times: decaying rate (surface filling up), unbinding uniform in time
    lo_f, hi_f, tau = 6, n_frames - 7, n_frames * 0.8          # truncated exponential arrivals
    u = rng.uniform(0, 1, n_b)
    fr_b = lo_f - tau * np.log(1 - u * (1 - np.exp(-(hi_f - lo_f) / tau)))
    fr_u = rng.uniform(6, n_frames - 7, n_u)
    frame = np.floor(np.concatenate([fr_b, fr_u])).astype(int)
    x = rng.uniform(3, nx - 4, n)
    y = rng.uniform(3, ny - 4, n)
    order = np.argsort(frame, kind="stable")
    contrast, frame, x, y = contrast[order], frame[order], x[order], y[order]
    pts = np.column_stack([x, y])
    # nearest neighbour distance among events in the same frame window (±5 frames)
    nn = np.full(n, 30.0)
    for i in range(n):
        near = np.abs(frame - frame[i]) <= 5
        near[i] = False
        if near.any():
            nn[i] = np.min(np.hypot(*(pts[near] - pts[i]).T))
    fit_err = np.clip(rng.lognormal(np.log(0.05), 0.6, n), 0.004, 0.37)

    # ---- movie: sum of FRAME_BINNING exposures, ~50000 counts, shot noise
    bg = 48000.0 * (1 + 0.05 * np.sin(np.linspace(0, np.pi, nx))[None, :] * np.cos(np.linspace(0, np.pi, ny))[:, None])
    psf = _psf()
    half = psf.shape[0] // 2
    landed = np.zeros((ny + 2 * half, nx + 2 * half))
    movie = np.empty((n_frames, ny, nx), np.uint16)
    j = 0
    for t in range(n_frames):
        while j < n and frame[j] <= t:
            cx, cy = int(round(x[j])) + half, int(round(y[j])) + half
            # AcquireMP convention: binding = negative contrast; raw pixels get brighter
            landed[cy - half:cy + half + 1, cx - half:cx + half + 1] += -contrast[j] * 1.2 * psf
            j += 1
        mean = bg * (1 + landed[half:half + ny, half:half + nx])
        movie[t] = np.clip(rng.normal(mean, np.sqrt(mean)), 0, 65535).astype(np.uint16)
    diff = np.empty_like(movie)
    diff[0] = movie[0]
    diff[1:] = movie[1:] - movie[:-1]                       # uint16 wrap-around = mod 2**16
    kf_idx = np.arange(0, n_frames, 100)

    dt = FRAME_BINNING / FRAME_RATE
    t0 = 4.09e12
    stack = t0 + (np.arange(n_frames)[:, None] * dt + np.arange(FRAME_BINNING)[None, :] / FRAME_RATE) * 1e9

    zstd = hdf5plugin.Zstd(clevel=3)
    with h5py.File(path, "w") as f:
        f["format_version_number"] = np.int32(4)
        mv = f.create_group("movie")
        d = mv.create_dataset("frame", data=diff, chunks=(1, ny, nx), shuffle=True, **zstd)
        d.attrs["encoding"] = "TIME_DIFFERENCE"
        k = mv.create_dataset("keyframe", data=movie[kf_idx], chunks=(1, ny, nx))
        k.attrs["indices"] = kf_idx
        mv["timeStack"] = stack
        mv["systemTimeStack"] = stack - 1.2e10
        mv["frameId"] = np.arange(n_frames, dtype=np.int32) + 1000
        mv["comments"] = comments
        mv["timestamp"] = timestamp
        mv["is_calibrant"] = bool(is_calibrant)
        mv["format_version_number"] = np.int32(20)
        mv["calibrant_info__is_none__"] = np.int32(0)
        mv["temperature__is_none__"] = np.int32(0)
        for k_, v in {"name": "mprfile demo generator", "publisher": "simulated", "version": "demo"}.items():
            mv[f"app_info/{k_}"] = v
        cam = {"exposure_time": 2.05, "frame_binning": np.int32(FRAME_BINNING), "frame_rate": FRAME_RATE,
               "image_size_name": "DEMO", "model": "simulated", "offset_x": np.int32(0),
               "offset_y": np.int32(0), "pixel_binning": np.int32(6)}
        for k_, v in cam.items():
            mv[f"configuration/acq_camera/{k_}"] = v
        mv["configuration/scatter_interference_destructive"] = False
        mv["configuration/format_version_number"] = np.int32(4)
        for k_, v in {"InstrumentName": "Simulated MP", "AcqCam": "SIM-0001", "InstrumentKey": "0"}.items():
            mv[f"device_serials/{k_}"] = v
        mi = {"sample": sample, "buffer": "simulated buffer", "bufferpH": 7.4, "operatorName": "demo"}
        for k_, v in mi.items():
            mv[f"measurement_info/{k_}"] = v
        mv["autofocus_image"] = rng.normal(0, 1, (33, 33))

        an = f.create_group("analysis")
        an["format_version_number"] = np.int32(14)
        an["is_analysed"] = True
        an["analysis_params/n_avg"] = np.int32(5)
        an["analysis_params/max_error"] = 0.375
        ev = an.create_group("events/events_fitted")
        ev["contrasts"] = contrast
        ev["x_coords"] = x
        ev["y_coords"] = y
        ev["frame_indices"] = frame.astype(float)
        ev["fit_errors"] = fit_err
        ev["residual_errors"] = fit_err * 5e-4
        ev["nearest_neighbour_distances"] = nn
        ev["ids"] = np.arange(n, dtype=np.uint64) + 1
        ev["good_flags"] = np.ones(n, bool)
        ev["selections"] = np.ones(n, bool)
        ev["calibrated_values"] = np.full(n, np.nan)
        ev["calibration__is_none__"] = np.int32(0)
        sc = an.create_group("scores")
        sc["brightness"] = 76.6 + rng.normal(0, 0.01, n_frames).cumsum() * 0.01
        sc["sharpness"] = 6.5 + rng.normal(0, 0.005, n_frames)
        sc["saturation"] = np.full(n_frames, 0.08)
    return os.fspath(path)


def make_demo_files(folder="demo_data", seed: int = 7) -> tuple[str, str]:
    """Write demo_ladder.mpr and demo_sample.mpr to `folder` and return their paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # calibrant: MassFerence-P1-like ladder, n x 86 kDa, decreasing abundance
    ladder = _species(rng, [(86 * k, c) for k, c in zip(range(1, 8), [640, 600, 500, 280, 140, 70, 35])])
    ladder = np.concatenate([ladder, rng.uniform(40, 800, 260)])
    ladder_unbind = _species(rng, [(86, 180), (172, 110)])
    p1 = write_demo_mpr(folder / "demo_ladder.mpr", ladder, ladder_unbind, sample="Demo ladder (simulated)",
                        comments="Simulated MassFerence-P1-like ladder", seed=seed, is_calibrant=True,
                        timestamp="2026_01_01-10_00_00")

    # sample: monomer 150, dimer 300, a minor trimer at 450 and a broad low-mass population
    sample = _species(rng, [(150, 1100), (300, 420), (450, 110)])
    sample = np.concatenate([sample, rng.normal(70, 18, 220), rng.uniform(40, 900, 180)])
    sample_unbind = _species(rng, [(150, 380), (300, 90)])
    p2 = write_demo_mpr(folder / "demo_sample.mpr", sample, sample_unbind, sample="Demo protein (simulated)",
                        comments="Simulated sample: 150/300/450 kDa", seed=seed + 1,
                        timestamp="2026_01_01-10_30_00")
    return p1, p2
