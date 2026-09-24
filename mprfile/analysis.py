"""
mprfile.analysis — from .mpr files to masses, peak tables and reports.

No-code use: see analyze_measurements.ipynb or the `mpr-analyze` command.

Python use:
    from mprfile import analyze, analyze_sample, calibrate, AnalysisSettings
    results = analyze(["019_252_50nM.mpr"], calibrant_file="002_Ladder.mpr",
                      calibrant="MassFerence P1", out="results")
    results.summary               # one row per peak, all samples

    cal = calibrate("002_Ladder.mpr", "MassFerence P1")
    r = analyze_sample("019_252_50nM.mpr", cal, AnalysisSettings(bin_width=4))
    r.peaks, r.events, r.warnings
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import Calibration, acquisition_info, calibrate, compare_acquisition
from .mixture import ROIResult, fit_rois
from .peaks import PeakFit, fit_peaks
from .reader import MPRFile

__all__ = ["AnalysisSettings", "SampleResult", "BatchResult", "analyze_sample", "analyze",
           "find_calibrant_file"]


@dataclass
class AnalysisSettings:
    """Settings for the sample analysis. Defaults suit a Refeyn TwoMP with MassFerence P1.

    mass_range      (kDa) window in which peaks are searched and fitted. The lower limit
                    keeps the fit away from the noise near zero mass.
    bin_width       (kDa) histogram bin width for fitting and plots.
    smooth_bins     smoothing (in bins) used only to *find* peaks, not for the fit.
    min_prominence  a peak must stand out by this fraction of the highest bin.
    min_counts      peaks with fewer fitted events are ignored.
    min_fraction    peaks with a smaller share of events in mass_range are ignored.
    sigma_max       (kDa) upper limit on a peak's width.
    max_fit_error   drop events whose AcquireMP PSF-fit error exceeds this (None = keep all
                    events AcquireMP marked good).
    background      fit one extra broad Gaussian under the peaks (for smeared samples).
    min_events      warn when a sample has fewer binding events than this.
    plot_max        (kDa) right edge of the histogram plot; None = automatic.

    Manual ROI mode (replaces the automatic peak search when `rois` is set):
    rois            list of (lo, hi, k): fit k Gaussians to the binding events in lo–hi kDa,
                    e.g. [(400, 620, 2), (40, 160, 1)]. Every ROI is also fitted with
                    1..roi_k_max Gaussians and compared by BIC; overfitting is flagged.
    roi_background  include a flat background in each ROI (recommended).
    roi_k_max       largest number of Gaussians tried for the model comparison.
    bootstrap       number of bootstrap refits per ROI for stability checks (0 = off; ~100).
    """
    mass_range: tuple[float, float] = (40.0, 1500.0)
    bin_width: float = 5.0
    smooth_bins: float = 2.0
    min_prominence: float = 0.05
    min_counts: int = 20
    min_fraction: float = 0.03
    sigma_max: float = 40.0
    max_fit_error: float | None = None
    background: bool = False
    min_events: int = 500
    plot_max: float | None = None
    rois: list | None = None
    roi_background: bool = True
    roi_k_max: int = 4
    bootstrap: int = 0


@dataclass
class SampleResult:
    file: str
    info: dict                      # sample name, buffer, timestamp, comments, instrument...
    events: pd.DataFrame            # all good events with a 'mass' column (kDa)
    peaks: pd.DataFrame             # fitted mass peaks with flags
    fit: PeakFit                    # histogram + Gaussian parameters (binding events)
    qc: dict                        # event counts, landing rate...
    warnings: list[str]
    calibration: Calibration
    settings: AnalysisSettings
    rois: list[ROIResult] | None = None     # manual ROI fits (None = automatic mode)

    @property
    def mode(self) -> str:
        """'automatic' or 'manual ROIs'."""
        return "manual ROIs" if self.rois else "automatic"

    @property
    def name(self) -> str:
        """File name without extension; used as the results sub-folder name."""
        return Path(self.file).stem

    def summary_rows(self) -> pd.DataFrame:
        """This sample's rows for summary.csv (one per peak, with sample/file/date columns)."""
        cols = {"sample": self.info.get("sample"), "file": os.path.basename(self.file),
                "measured": self.info.get("timestamp"),
                "binding_events": self.qc["binding_events"]}
        if not len(self.peaks):
            return pd.DataFrame([{**cols, "peak": None}])
        df = self.peaks.copy()
        for k, v in reversed(cols.items()):
            df.insert(0, k, v)
        return df


@dataclass
class BatchResult:
    calibration: Calibration
    samples: list[SampleResult] = field(default_factory=list)
    out: str | None = None

    @property
    def summary(self) -> pd.DataFrame:
        """One row per peak for all samples (what summary.csv contains)."""
        if not self.samples:
            return pd.DataFrame()
        return pd.concat([s.summary_rows() for s in self.samples], ignore_index=True)

    def __repr__(self):
        return (f"BatchResult({len(self.samples)} sample(s), calibration={self.calibration!r}, "
                f"out={self.out!r})")


# ---------------------------------------------------------------------------- one sample
def analyze_sample(path, calibration: Calibration, settings: AnalysisSettings | None = None) -> SampleResult:
    """Convert one sample's events to mass and fit its mass peaks."""
    st = settings or AnalysisSettings()
    with MPRFile(path) as m:
        ev = m.events_df(calibration=calibration)
        info = {**acquisition_info(m), **m.sample_info}
        duration = float(m.times()[-1])

    warnings = list(compare_acquisition(calibration.source, info)) if calibration.source else []
    if st.max_fit_error is not None:
        n0 = len(ev)
        ev = ev[ev.fit_error <= st.max_fit_error].reset_index(drop=True)
        info["events_removed_by_fit_error"] = n0 - len(ev)

    binding = ev[ev.mass > 0]
    n_range = int(((binding.mass >= st.mass_range[0]) & (binding.mass <= st.mass_range[1])).sum())
    roi_results = None
    if st.rois:
        roi_results = fit_rois(binding.mass.values, st.rois, resolution=calibration.resolution_kDa,
                               k_max=st.roi_k_max, background=st.roi_background, bootstrap=st.bootstrap)
        peaks = roi_peaks_table(roi_results, n_range, calibration)
        fit = PeakFit(pd.DataFrame(), np.array([0.0, 1.0]), np.array([0]), np.array([]), None, n_range,
                      ["manual ROI mode"])
    else:
        fit = fit_peaks(binding.mass.values, st.mass_range[0], st.mass_range[1], st.bin_width,
                        smooth_bins=st.smooth_bins, min_prominence=st.min_prominence,
                        min_counts=st.min_counts, min_fraction=st.min_fraction,
                        sigma_guess=max(3 * st.bin_width, 10.0), sigma_max=st.sigma_max,
                        background=st.background)
        peaks = fit.peaks.rename(columns={"position": "mass_kDa", "position_err": "mass_err_kDa",
                                          "sigma": "sigma_kDa", "fraction": "fraction_in_range"})
        peaks.insert(0, "peak", np.arange(1, len(peaks) + 1))
        peaks["percent"] = 100 * peaks.pop("fraction_in_range")
        peaks["notes"] = [_notes(p, calibration, st) for p in peaks.itertuples()]

    qc = {
        "events": int(len(ev)),
        "binding_events": int((ev.mass > 0).sum()),
        "unbinding_events": int((ev.mass < 0).sum()),
        "binding_in_mass_range": n_range,
        "assigned_to_peaks_pct": round(float(peaks.percent.sum()), 1) if len(peaks) else 0.0,
        "duration_s": round(duration, 1),
        "landing_rate_first_10s": round(float(((ev.mass > 0) & (ev.time < 10)).sum() / 10), 1),
    }
    if qc["binding_events"] < st.min_events:
        warnings.append(f"Only {qc['binding_events']} binding events (< {st.min_events}); "
                        "peak positions and percentages are less reliable")
    if roi_results:
        for r in roi_results:
            warnings += [f"ROI {r.label}: {w}" for w in r.warnings]
        for p in peaks.itertuples():
            rng_note = _range_note(p.mass_kDa, calibration)
            if rng_note:
                warnings.append(f"Peak {p.peak} ({p.mass_kDa:.0f} kDa): {rng_note}")
    else:
        for p in peaks.itertuples():
            if p.notes:
                warnings.append(f"Peak {p.peak} ({p.mass_kDa:.0f} kDa): {p.notes}")
        warnings += fit.notes
    return SampleResult(os.fspath(path), info, ev, peaks, fit, qc, warnings, calibration, st, roi_results)


def roi_peaks_table(roi_results: list[ROIResult], n_range: int, cal: Calibration) -> pd.DataFrame:
    """Peak table (same columns as automatic mode) from manual ROI fits, with per-peak notes
    taken from the ROI diagnostics."""
    rows = []
    for r in roi_results:
        f = r.fit
        for i in range(f.k):
            tag = f"component at {f.mu[i]:.0f} kDa"
            notes = [w.split(": ", 1)[1] for w in r.warnings if w.startswith(tag)]
            notes += [w for w in r.warnings if w.startswith(("OVERFITTING", "UNDERFITTING"))]
            notes += [w for w in r.warnings if "not resolved" in w and f"{f.mu[i]:.0f} " in w + " "]
            rn = _range_note(f.mu[i], cal)
            if rn:
                notes.insert(0, rn)
            row = {"mass_kDa": f.mu[i],
                   "mass_err_kDa": f.mu_err[i] if f.mu_err is not None else np.nan,
                   "sigma_kDa": f.sigma[i], "counts": f.counts[i],
                   "percent": 100 * f.counts[i] / max(n_range, 1),
                   "roi": r.label, "roi_k": r.k, "roi_bic_best_k": r.bic_best_k, "roi_risk": r.risk,
                   "notes": "; ".join(dict.fromkeys(notes))}
            if r.bootstrap is not None:
                row["mass_sd_bootstrap"] = r.bootstrap.mass_sd.values[i]
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("mass_kDa", ignore_index=True)
        df.insert(0, "peak", np.arange(1, len(df) + 1))
    return df


def _range_note(mass: float, cal: Calibration) -> str:
    rng = cal.mass_range
    if rng:
        if mass < 0.9 * rng[0]:
            return f"below calibrated range ({rng[0]:.0f}–{rng[1]:.0f} kDa)"
        if mass > 1.1 * rng[1]:
            return f"above calibrated range ({rng[0]:.0f}–{rng[1]:.0f} kDa), extrapolated"
    return ""


def _notes(p, cal: Calibration, st: AnalysisSettings) -> str:
    out = [n for n in [_range_note(p.mass_kDa, cal)] if n]
    if p.sigma_kDa >= 0.98 * st.sigma_max:
        out.append(f"width hit the limit (σ = {st.sigma_max:.0f} kDa): not a single clean peak, "
                   "mass and % are approximate")
    if p.mass_kDa - 1.5 * p.sigma_kDa < st.mass_range[0]:
        out.append(f"cut off by lower mass limit ({st.mass_range[0]:.0f} kDa)")
    if p.mass_kDa + 1.5 * p.sigma_kDa > st.mass_range[1]:
        out.append(f"cut off by upper mass limit ({st.mass_range[1]:.0f} kDa)")
    return "; ".join(out)


# ---------------------------------------------------------------------------- batch
CALIBRANT_NAME_PATTERN = re.compile(r"ladder|calib|massference|mpf|standard", re.I)


def find_calibrant_file(files) -> str | None:
    """Pick the calibrant among files: flagged as calibrant in AcquireMP, or with a
    sample/file name containing ladder/calib/massference/mpf/standard."""
    hits = []
    for f in files:
        try:
            with MPRFile(f) as m:
                flagged = bool(m.h5["movie/is_calibrant"][()]) if "movie/is_calibrant" in m.h5 else False
                name = str(m.sample_info.get("sample", ""))
        except OSError:
            continue
        if flagged or CALIBRANT_NAME_PATTERN.search(name) or CALIBRANT_NAME_PATTERN.search(Path(f).stem):
            hits.append(f)
    return hits[0] if len(hits) == 1 else None


def _expand(samples) -> list[str]:
    if samples is None:
        samples = "*.mpr"
    if isinstance(samples, (str, os.PathLike)):
        samples = [samples]
    files = []
    for s in samples:
        s = os.fspath(s)
        if os.path.isdir(s):
            files += sorted(glob.glob(os.path.join(s, "*.mpr")))
        elif any(ch in s for ch in "*?["):
            files += sorted(glob.glob(s))
        else:
            files.append(s)
    seen, out = set(), []
    for f in files:
        k = os.path.abspath(f)
        if k not in seen:
            seen.add(k); out.append(f)
    return out


def analyze(samples=None, *, calibrant_file=None, calibrant="MassFerence P1",
            calibration: Calibration | str | None = None, out="results",
            settings: AnalysisSettings | None = None, reports: bool = True,
            verbose: bool = True) -> BatchResult:
    """Calibrate once and analyse all samples, writing everything to `out`.

    samples:        files, folders or wildcards (default: all *.mpr in the current folder).
    calibrant_file: the calibrant measurement; if None it is picked automatically from the
                    files (name contains 'ladder', 'calib', ...), unless `calibration` is given.
    calibrant:      calibrant name (e.g. "MassFerence P1") or list of masses in kDa.
    calibration:    an existing Calibration or a saved calibration .json (skips calibrant).
    out:            output folder (None = don't write files).
    """
    st = settings or AnalysisSettings()
    files = _expand(samples)
    log = print if verbose else (lambda *a, **k: None)

    if isinstance(calibration, (str, os.PathLike)):
        cal = Calibration.load(calibration)
        log(f"Loaded calibration from {calibration}")
    elif isinstance(calibration, Calibration):
        cal = calibration
    else:
        if calibrant_file is None:
            calibrant_file = find_calibrant_file(files)
            if calibrant_file is None:
                raise ValueError("Could not tell which file is the calibrant. Set calibrant_file "
                                 "(e.g. calibrant_file='002_Ladder.mpr').")
            log(f"Calibrant file (auto-detected): {calibrant_file}")
        cal = calibrate(calibrant_file, calibrant)
        log(repr(cal))
        for w in cal.warnings:
            log(f"  ! {w}")

    cal_path = os.path.abspath(calibrant_file) if calibrant_file else None
    files = [f for f in files if os.path.abspath(f) != cal_path]
    if not files:
        log("No sample files to analyse.")
    res = BatchResult(cal, out=os.fspath(out) if out else None)
    for f in files:
        r = analyze_sample(f, cal, st)
        res.samples.append(r)
        pk = ", ".join(f"{p.mass_kDa:.0f} kDa ({p.percent:.0f}%)" for p in r.peaks.itertuples()) or "no peaks"
        log(f"{os.path.basename(f)}: {r.qc['binding_events']} binding events -> {pk}")
        for w in r.warnings:
            log(f"  ! {w}")

    if out:
        write_results(res, out, reports=reports)
        log(f"Results written to {os.path.abspath(out)}")
    return res


def write_results(res: BatchResult, out, reports: bool = True) -> None:
    out = Path(out)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    cal = res.calibration
    cal.save(out / "calibration" / "calibration.json")
    if cal.peaks is not None:
        cal.peaks.to_csv(out / "calibration" / "calibration_peaks.csv", index=False)
    if cal.validation is not None:
        cal.validation.to_csv(out / "calibration" / "ladder_check.csv", index=False)

    for r in res.samples:
        write_sample(r, out, reports=False)
    if res.samples:
        res.summary.to_csv(out / "summary.csv", index=False)
    pd.Series({k: (str(v) if isinstance(v, (tuple, list)) else v) for k, v in
               asdict(res.samples[0].settings if res.samples else AnalysisSettings()).items()}
              ).to_csv(out / "settings_used.csv", header=["value"])

    if reports:
        from .report import write_reports
        write_reports(res, out)


def write_sample(r: SampleResult, out, reports: bool = True) -> Path:
    """Write one sample's CSVs (and report) to out/<sample>/."""
    d = Path(out) / r.name
    d.mkdir(parents=True, exist_ok=True)
    r.peaks.to_csv(d / "peaks.csv", index=False)
    r.events.to_csv(d / "events.csv", index=False)
    pd.Series({**r.qc, **{k: v for k, v in r.info.items() if k in
               ("sample", "buffer", "bufferpH", "timestamp", "comments")}}).to_csv(
        d / "info.csv", header=["value"])
    wfile = d / "warnings.txt"
    if r.warnings:
        wfile.write_text("\n".join(r.warnings) + "\n", encoding="utf-8")
    elif wfile.exists():
        wfile.unlink()
    for f in ("rois.json", "roi_model_comparison.csv"):
        if not r.rois and (d / f).exists():
            (d / f).unlink()
    if r.rois:
        write_roi_files(r.rois, d)
    if reports:
        import matplotlib.pyplot as plt
        from .report import plot_sample
        with plt.ioff():
            fig = plot_sample(r)
        fig.savefig(d / "report.pdf")
        fig.savefig(d / "report.png", dpi=130)
        plt.close(fig)
    return d


def update_summary(r: SampleResult, out) -> None:
    """Replace this sample's rows in out/summary.csv (or create it)."""
    f = Path(out) / "summary.csv"
    new = r.summary_rows()
    if f.exists():
        old = pd.read_csv(f)
        old = old[old["file"] != os.path.basename(r.file)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(f, index=False)


def write_roi_files(rois: list[ROIResult], folder) -> None:
    """Model comparison tables and a reproducible ROI definition."""
    import json
    folder = Path(folder)
    tabs = []
    for r in rois:
        t = r.models.copy()
        t.insert(0, "roi", r.label)
        t["chosen"] = t.k == r.k
        tabs.append(t)
    pd.concat(tabs, ignore_index=True).to_csv(folder / "roi_model_comparison.csv", index=False)
    spec = {"rois": [list(r.spec()) for r in rois], "background": rois[0].background,
            "model_check": [r.model_summary() for r in rois],
            "risk": {r.label: r.risk for r in rois},
            "warnings": {r.label: r.warnings for r in rois},
            "reproduce": "AnalysisSettings(rois=" + repr([r.spec() for r in rois]) + ")"}
    (folder / "rois.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
