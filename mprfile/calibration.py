"""
mprfile.calibration — contrast-to-mass calibration from a calibrant measurement.

    from mprfile import calibrate
    cal = calibrate("002_Ladder.mpr", "MassFerence P1")   # automatic peak matching
    print(cal.report())
    cal.save("calibration.json");  cal = Calibration.load("calibration.json")
    masses = cal(contrasts)

Mass photometry contrast is proportional to mass, so the calibration is a straight
line: mass = slope · contrast + intercept (slope < 0: binding events have negative
contrast in AcquireMP's convention and positive mass).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .peaks import PeakFit, fit_peaks
from .reader import MPRFile

__all__ = ["CALIBRANTS", "Calibration", "calibrate", "get_calibrant"]

# Known calibrants. `masses` are the certified peaks used for the fit; for an
# oligomer ladder, `unit_mass` lets extra (higher) oligomer peaks be used as an
# independent check of linearity.
CALIBRANTS: dict[str, dict] = {
    "MassFerence P1": {
        "masses": [86.0, 172.0, 258.0, 344.0],
        "unit_mass": 86.0,
        "aliases": ["massference p1", "mpf1", "mp1", "p1", "massference"],
        "source": "Refeyn MassFerence P1 product datasheet (July 2025): 86, 172, 258, 344 kDa",
    },
}


def get_calibrant(name_or_masses) -> tuple[str, list[float], float | None]:
    """Resolve a calibrant name (case-insensitive, aliases allowed) or a list of
    masses in kDa. Returns (name, masses, unit_mass)."""
    if isinstance(name_or_masses, str):
        key = name_or_masses.strip().lower()
        for name, spec in CALIBRANTS.items():
            if key == name.lower() or key in spec["aliases"]:
                return name, list(spec["masses"]), spec.get("unit_mass")
        # "86,172,258" typed as a string
        try:
            masses = [float(v) for v in name_or_masses.replace(";", ",").split(",") if v.strip()]
            return "custom", masses, None
        except ValueError:
            raise ValueError(f"Unknown calibrant {name_or_masses!r}. Known: {list(CALIBRANTS)} "
                             f"or give the masses in kDa, e.g. [66, 132, 480]") from None
    masses = [float(v) for v in name_or_masses]
    return "custom", masses, None


@dataclass
class Calibration:
    """Linear calibration  mass = slope · contrast + intercept  (masses in kDa)."""
    slope: float
    intercept: float = 0.0
    unit: str = "kDa"
    r2: float | None = None
    calibrant: str | None = None
    peaks: pd.DataFrame | None = None           # matched calibrant peaks + residuals
    validation: pd.DataFrame | None = None      # extra ladder peaks (not used in the fit)
    source: dict = field(default_factory=dict)  # calibrant file, date, instrument, settings
    warnings: list[str] = field(default_factory=list)
    peakfit: PeakFit | None = field(default=None, repr=False)  # calibrant histogram fit (plots)

    # --- construction ------------------------------------------------------
    @classmethod
    def fit(cls, contrasts, masses, unit: str = "kDa", **kw) -> "Calibration":
        """Least-squares line through known (contrast, mass) pairs."""
        c, m = np.asarray(contrasts, float), np.asarray(masses, float)
        if len(c) < 2:
            raise ValueError("need at least two calibration points")
        slope, intercept = np.polyfit(c, m, 1)
        pred = slope * c + intercept
        r2 = 1 - np.sum((m - pred) ** 2) / np.sum((m - m.mean()) ** 2) if len(c) > 2 else 1.0
        return cls(float(slope), float(intercept), unit, float(r2), **kw)

    # --- use ---------------------------------------------------------------
    def __call__(self, contrast):
        return self.slope * np.asarray(contrast, float) + self.intercept

    def inverse(self, mass):
        """Contrast corresponding to a mass (the inverse calibration)."""
        return (np.asarray(mass, float) - self.intercept) / self.slope

    @property
    def mass_range(self) -> tuple[float, float] | None:
        """Mass range supported by calibrant peaks (fitted + validated ladder peaks)."""
        ms = []
        for df in (self.peaks, self.validation):
            if df is not None and len(df):
                ok = df["used"] if "used" in df else df["within_5pct"]
                ms += list(df.loc[ok, "known_mass"])
        return (min(ms), max(ms)) if ms else None

    @property
    def resolution_kDa(self) -> float | None:
        """Typical width (σ, kDa) of a single species: median width of the calibrant peaks.
        Used to flag fitted components that are implausibly narrow."""
        if self.peaks is not None and "sigma_kDa" in self.peaks and len(self.peaks):
            return float(np.median(self.peaks["sigma_kDa"]))
        return None

    def __repr__(self):
        r2 = f", R²={self.r2:.5f}" if self.r2 is not None else ""
        name = f" [{self.calibrant}]" if self.calibrant else ""
        return f"Calibration(mass = {self.slope:.6g}·contrast {self.intercept:+.4g} {self.unit}{r2}){name}"

    def report(self) -> str:
        """Text report: formula, calibrant peaks with residuals, ladder check, range, warnings."""
        lines = [repr(self)]
        if self.source:
            lines.append(f"Calibrant file: {self.source.get('file')}  ({self.source.get('timestamp')})")
        if self.peaks is not None:
            lines.append("\nCalibration peaks:")
            lines.append(self.peaks.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
        if self.validation is not None and len(self.validation):
            lines.append("\nExtra ladder peaks (not fitted, independent check):")
            lines.append(self.validation.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
        if self.mass_range:
            lines.append(f"\nSupported mass range: {self.mass_range[0]:.0f}–{self.mass_range[1]:.0f} kDa")
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)

    # --- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        """Plain-dict form of the calibration (what save() writes)."""
        d = {"slope": self.slope, "intercept": self.intercept, "unit": self.unit, "r2": self.r2,
             "calibrant": self.calibrant, "source": self.source, "warnings": self.warnings,
             "created": _dt.datetime.now().isoformat(timespec="seconds"),
             "formula": "mass = slope * contrast + intercept"}
        for k in ("peaks", "validation"):
            df = getattr(self, k)
            d[k] = None if df is None else json.loads(df.to_json(orient="records"))
        return d

    def save(self, path) -> str:
        """Save as JSON (formula, peaks, source file and settings) for reuse with later samples."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return os.fspath(path)

    @classmethod
    def load(cls, path) -> "Calibration":
        """Load a calibration saved with save() (a calibration.json from the results folder)."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(slope=d["slope"], intercept=d["intercept"], unit=d.get("unit", "kDa"), r2=d.get("r2"),
                   calibrant=d.get("calibrant"),
                   peaks=pd.DataFrame(d["peaks"]) if d.get("peaks") else None,
                   validation=pd.DataFrame(d["validation"]) if d.get("validation") else None,
                   source=d.get("source", {}), warnings=d.get("warnings", []))


def _match(peak_c: np.ndarray, peak_n: np.ndarray, masses: np.ndarray, tol: float):
    """Assign detected peaks (|contrast|) to known masses assuming mass ∝ contrast.
    Tries every (peak, mass) pair as anchor and keeps the assignment matching the most
    known masses (ties: most events). Returns (pairs, ambiguous) where pairs is a list of
    (peak_index, mass_index) and ambiguous is True when a clearly different scale matches
    equally many masses (e.g. a 2-mass calibrant on a ladder)."""
    candidates = {}
    for c0 in peak_c:
        for m0 in masses:
            scale = m0 / c0
            pred = peak_c * scale
            pairs, used = [], set()
            for j, mj in enumerate(masses):
                i = int(np.argmin(np.abs(pred - mj)))
                if abs(pred[i] - mj) <= tol * mj and i not in used:
                    pairs.append((i, j)); used.add(i)
            key = tuple(sorted(pairs, key=lambda p: p[1]))
            if key:
                score = (len(key), float(sum(peak_n[i] for i, _ in key)))
                candidates[key] = score
    if not candidates:
        return [], False
    best = max(candidates, key=candidates.get)
    n_best = candidates[best][0]

    def _scale(pairs):
        return np.median([masses[j] / peak_c[i] for i, j in pairs])
    s_best = _scale(best)
    ambiguous = any(sc[0] == n_best and abs(_scale(k) / s_best - 1) > 0.1
                    for k, sc in candidates.items() if k != best)
    return list(best), ambiguous


def calibrate(source, calibrant="MassFerence P1", *, contrast_range=(0.0008, 0.05),
              bin_width: float = 1e-4, tolerance: float = 0.08, max_fit_error: float | None = None,
              min_peak_counts: int = 100) -> Calibration:
    """Build a Calibration from a calibrant .mpr file (path or open MPRFile).

    calibrant: a name from CALIBRANTS (e.g. "MassFerence P1") or a list of masses in kDa.
    contrast_range: |contrast| window searched for binding peaks.
    tolerance: max relative mismatch when assigning peaks to known masses (0.08 = 8%).
    max_fit_error: optionally drop events with AcquireMP fit_error above this value.
    """
    name, masses, unit_mass = get_calibrant(calibrant)
    masses = np.array(sorted(masses), float)
    own = not isinstance(source, MPRFile)
    m = MPRFile(source) if own else source
    try:
        ev = m.events()
        info = acquisition_info(m)
    finally:
        if own:
            m.close()
    sel = np.ones(len(ev["contrast"]), bool)
    if max_fit_error is not None:
        sel &= ev["fit_error"] <= max_fit_error
    neg = -ev["contrast"][sel]                       # binding events -> positive values

    pf = fit_peaks(neg, contrast_range[0], contrast_range[1], bin_width, smooth_bins=3,
                   min_prominence=0.03, min_counts=15, min_fraction=0.005,
                   sigma_guess=5 * bin_width, sigma_max=20 * bin_width, background=True)
    if len(pf.peaks) < 2:
        raise RuntimeError(f"Only {len(pf.peaks)} peak(s) found in the calibrant file; cannot calibrate.")
    pc, pn = pf.peaks.position.values, pf.peaks.counts.values
    pairs, ambiguous = _match(pc, pn, masses, tolerance)
    if len(pairs) < 2:
        raise RuntimeError(f"Could not match the calibrant peaks to {name} masses {list(masses)} kDa. "
                           f"Detected |contrast| peaks: {np.round(pc, 5).tolist()}. "
                           "Check the calibrant name/masses or pass them explicitly.")

    ip = [i for i, _ in pairs]
    known = masses[[j for _, j in pairs]]
    contrast = -pc[ip]                                # back to signed (negative) contrast
    cal = Calibration.fit(contrast, known, calibrant=name, source=info, peakfit=pf)

    fitted = cal(contrast)
    cal.peaks = pd.DataFrame({
        "known_mass": known, "contrast": contrast, "contrast_err": pf.peaks.position_err.values[ip],
        "counts": pn[ip], "fitted_mass": fitted, "error_kDa": fitted - known,
        "error_pct": 100 * (fitted - known) / known, "used": True,
        "sigma_kDa": pf.peaks.sigma.values[ip] * abs(cal.slope)})

    # extra ladder peaks: independent check of linearity beyond the certified peaks
    if unit_mass:
        rows = []
        for i in set(range(len(pc))) - set(ip):
            mass = float(cal(-pc[i]))
            n = int(round(mass / unit_mass))
            if n >= 1:
                exp = n * unit_mass
                rows.append({"oligomer": n, "known_mass": exp, "contrast": -pc[i], "counts": pn[i],
                             "fitted_mass": mass, "error_kDa": mass - exp,
                             "error_pct": 100 * (mass - exp) / exp})
        if rows:
            v = pd.DataFrame(rows).sort_values("known_mass", ignore_index=True)
            v["within_5pct"] = v.error_pct.abs() <= 5
            cal.validation = v

    # quality checks
    w = cal.warnings
    if ambiguous:
        w.append("Peak assignment is ambiguous (another set of peaks fits these masses equally well); "
                 "check in the calibration report that the right peaks were used, or give more masses")
    lo_c, hi_c = pc[ip].min(), pc[ip].max()
    stray = [i for i in range(len(pc)) if i not in ip and lo_c < pc[i] < hi_c and pn[i] >= min_peak_counts]
    if stray:
        w.append(f"{len(stray)} strong peak(s) between the calibration peaks are not assigned "
                 f"(at {[round(float(cal(-pc[i]))) for i in stray]} kDa); the calibrant name/masses may be wrong")
    missing = sorted(set(masses) - set(known))
    if missing:
        w.append(f"Calibrant peaks not found: {missing} kDa")
    if len(known) < 3:
        w.append(f"Only {len(known)} calibrant peaks matched; ≥3 recommended")
    low = cal.peaks.loc[cal.peaks.counts < min_peak_counts, "known_mass"].tolist()
    if low:
        w.append(f"Calibrant peak(s) {low} kDa have <{min_peak_counts} counts")
    if cal.r2 is not None and cal.r2 < 0.999:
        w.append(f"Calibration linearity R² = {cal.r2:.4f} (< 0.999)")
    worst = cal.peaks.error_pct.abs().max()
    if worst > 5:
        w.append(f"Largest calibrant residual {worst:.1f}% (> 5%)")
    if cal.validation is not None and (~cal.validation.within_5pct).any():
        bad = cal.validation.loc[~cal.validation.within_5pct, "known_mass"].tolist()
        w.append(f"Extra ladder peak(s) at {bad} kDa deviate >5% from linearity")
    return cal


def acquisition_info(m: MPRFile) -> dict:
    """Instrument and acquisition settings that must match between calibrant and sample."""
    cam = m.camera
    ins = m.instrument
    s = m.sample_info
    return {
        "file": os.path.basename(m.path),
        "timestamp": s.get("timestamp"),
        "sample": s.get("sample"),
        "instrument": ins.get("InstrumentName"),
        "instrument_serial": ins.get("AcqCam"),
        "image_size": cam.get("image_size_name"),
        "frame_binning": cam.get("frame_binning"),
        "pixel_binning": cam.get("pixel_binning"),
        "frame_rate": round(float(cam.get("frame_rate", np.nan)), 3),
        "exposure_time": round(float(cam.get("exposure_time", np.nan)), 4),
        "n_avg": m.analysis_params.get("n_avg"),
        "software": _software(m),
    }


def _software(m: MPRFile) -> str | None:
    try:
        v = m.h5["movie/app_info/version"][()]
        return v.decode() if isinstance(v, bytes) else str(v)
    except KeyError:
        return None


def compare_acquisition(cal_info: dict, sample_info: dict) -> list[str]:
    """Warnings for settings that differ between calibrant and sample measurement."""
    out = []
    for key, label in [("instrument_serial", "instrument"), ("image_size", "image size"),
                       ("frame_binning", "frame binning"), ("pixel_binning", "pixel binning"),
                       ("frame_rate", "frame rate"), ("n_avg", "analysis n_avg")]:
        a, b = cal_info.get(key), sample_info.get(key)
        if a is not None and b is not None and a != b:
            out.append(f"Calibrant and sample differ in {label} ({a} vs {b}); calibration may not apply")
    try:
        fmt = "%Y_%m_%d-%H_%M_%S"
        t0 = _dt.datetime.strptime(cal_info["timestamp"], fmt)
        t1 = _dt.datetime.strptime(sample_info["timestamp"], fmt)
        if abs((t1 - t0).days) >= 1:
            out.append(f"Calibrant measured {abs((t1 - t0).days)} day(s) from the sample; "
                       "a same-day calibration is recommended")
    except (KeyError, TypeError, ValueError):
        pass
    return out
