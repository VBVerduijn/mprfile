"""
mprfile.report — one-page PDF/PNG reports for the calibration and each sample.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .peaks import gauss

__all__ = ["plot_calibration", "plot_sample", "write_reports"]

C_DATA = "#3b6ea5"      # binding events
C_UNBIND = "#a9b4c0"    # unbinding events (muted)
C_FIT = "#333333"       # fitted curves
C_CHECK = "#d9722e"     # independent ladder check / flags
INK2 = "#555555"
A4 = (11.69, 8.27)

_STYLE = {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
          "grid.alpha": 0.25, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
          "axes.labelcolor": "#222222", "xtick.color": INK2, "ytick.color": INK2}


def _header(fig, title, subtitle):
    fig.text(0.04, 0.955, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.04, 0.915, subtitle, fontsize=9, color=INK2, va="top")


def _text_block(ax, lines, title=None, wrap=62, size=8.5, y0=1.0):
    """Write lines top-down with a fixed line height (independent of the axes size)."""
    ax.axis("off")
    h_in = ax.get_position().height * ax.figure.get_figheight()
    dy = (size * 1.45 / 72) / h_in
    y = y0
    if title:
        ax.text(0, y, title, fontsize=10, fontweight="bold", va="top", transform=ax.transAxes)
        y -= dy * 1.6
    for line, color in lines:
        for i, part in enumerate(textwrap.wrap(line, wrap) or [""]):
            ax.text(0.0 if i == 0 else 0.03, y, part, fontsize=size, va="top", color=color,
                    transform=ax.transAxes)
            y -= dy
        y -= dy * 0.25
    return y


def _table(ax, df, col_labels, fmt, title=None, widths=None, highlight=None, size=8.5):
    """Simple ruled table at the top of `ax`; returns the axes-fraction y below it."""
    ax.axis("off")
    h_in = ax.get_position().height * ax.figure.get_figheight()
    row = (size * 2.0 / 72) / h_in
    y_top = 1.0
    if title:
        ax.text(0, y_top, title, fontsize=10, fontweight="bold", va="top", transform=ax.transAxes)
        y_top -= row * 1.2
    cells = [[f(v) for f, v in zip(fmt, r)] for r in df.itertuples(index=False)]
    if not cells:
        ax.text(0, y_top, "none found", fontsize=size, va="top", transform=ax.transAxes, color=INK2)
        return y_top - row
    h = row * (len(cells) + 1)
    widths = widths or [1 / len(col_labels)] * len(col_labels)
    t = ax.table(cellText=cells, colLabels=col_labels, colWidths=widths, loc="upper left",
                 cellLoc="right", colLoc="right", bbox=[0, y_top - h, 1, h])
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.visible_edges = "B"
        cell.PAD = 0.04
        if r == 0:
            cell.set_text_props(fontweight="bold", color="#222222")
        elif highlight is not None and highlight[r - 1]:
            cell.set_text_props(color=C_CHECK)
    return y_top - h


# ---------------------------------------------------------------------------- calibration
def plot_calibration(cal):
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=A4)
        src = cal.source or {}
        _header(fig, f"Mass calibration: {cal.calibrant or 'custom'}",
                f"Calibrant file {src.get('file', '?')}  ·  measured {src.get('timestamp', '?')}  ·  "
                f"{src.get('instrument', '')}  ·  mass = {cal.slope:.1f} × contrast {cal.intercept:+.2f} kDa  ·  "
                f"R² = {cal.r2:.5f}")
        gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.85, bottom=0.07,
                              hspace=0.45, wspace=0.22, height_ratios=[1, 1])

        # histogram of calibrant contrast with matched peaks
        ax = fig.add_subplot(gs[0, :])
        pf = cal.peakfit
        if pf is not None:
            ax.stairs(pf.counts, -pf.edges, fill=True, color=C_DATA, alpha=0.85, label="calibrant events")
            xx = np.linspace(pf.edges[0], pf.edges[-1], 3000)
            ax.plot(-xx, pf.curve(xx), color=C_FIT, lw=1, label="Gaussian fit")
            allc = list(cal.peaks.contrast) + (list(cal.validation.contrast) if cal.validation is not None else [])
            ax.set_xlim(min(min(allc) * 1.2, -pf.edges[0] - 1e-3), 0)
        ymax = ax.get_ylim()[1]
        for p in cal.peaks.itertuples():
            ax.annotate(f"{p.known_mass:.0f}", (p.contrast, ymax * 0.93), ha="center", fontsize=8.5,
                        fontweight="bold", color=C_FIT)
        if cal.validation is not None:
            for p in cal.validation.itertuples():
                ax.annotate(f"{p.known_mass:.0f}", (p.contrast, ymax * 0.93), ha="center", fontsize=8.5,
                            color=C_CHECK)
        ax.set(xlabel="Contrast (binding events)   ·   labels: known mass in kDa; black = used for "
                      "calibration, orange = extra ladder peaks (check only)", ylabel="Events per bin",
               title="Calibrant contrast histogram")
        ax.set_ylim(0, ymax * 1.05)

        # calibration line
        ax = fig.add_subplot(gs[1, 0])
        allc = list(cal.peaks.contrast) + (list(cal.validation.contrast) if cal.validation is not None else [])
        cc = np.linspace(min(allc) * 1.08, 0, 50)
        ax.plot(cc, cal(cc), color=C_FIT, lw=1, label="calibration line")
        ax.plot(cal.peaks.contrast, cal.peaks.known_mass, "o", color=C_DATA, ms=7, label="calibrant peaks (fitted)")
        if cal.validation is not None and len(cal.validation):
            ax.plot(cal.validation.contrast, cal.validation.known_mass, "D", mfc="white", mec=C_CHECK,
                    mew=1.5, ms=6, label="extra ladder peaks (check)")
        ax.set(xlabel="Contrast", ylabel="Mass (kDa)", title="Contrast → mass")
        ax.legend(frameon=False, fontsize=8, loc="upper right")

        # residual table + warnings
        ax = fig.add_subplot(gs[1, 1])
        tab = cal.peaks[["known_mass", "contrast", "counts", "fitted_mass", "error_pct"]].copy()
        tab["kind"] = "fit"
        if cal.validation is not None and len(cal.validation):
            v = cal.validation[["known_mass", "contrast", "counts", "fitted_mass", "error_pct"]].copy()
            v["kind"] = "check"
            tab = pd.concat([tab, v], ignore_index=True)
        tab = tab.sort_values("known_mass")
        f0 = lambda v: f"{v:.0f}"
        y = _table(ax, tab, ["known kDa", "contrast", "events", "measured kDa", "error %", "role"],
                   [f0, lambda v: f"{v:.5f}", f0, lambda v: f"{v:.1f}", lambda v: f"{v:+.1f}", str],
                   title="Peaks", widths=[0.15, 0.18, 0.13, 0.23, 0.16, 0.15],
                   highlight=list(tab.kind == "check"))
        rng = cal.mass_range
        lines = [(f"Supported mass range: {rng[0]:.0f}–{rng[1]:.0f} kDa "
                  "(fitted + checked ladder peaks)", "#222222")] if rng else []
        lines += [(f"⚠ {w}", C_CHECK) for w in cal.warnings] or [("✓ No calibration warnings", "#2e7d32")]
        _text_block(ax, lines, wrap=85, y0=y - 0.06)
        return fig


# ---------------------------------------------------------------------------- sample
def _auto_plot_max(r):
    st = r.settings
    if st.plot_max:
        return st.plot_max
    m = r.events.mass[r.events.mass > 0]
    top = np.percentile(m, 99) if len(m) else 500
    if len(r.peaks):
        top = max(top, (r.peaks.mass_kDa + 3 * r.peaks.sigma_kDa).max())
    return float(min(max(np.ceil(top * 1.1 / 100) * 100, 200), st.mass_range[1]))


def plot_sample(r):
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=A4)
        info = r.info
        title = f"{info.get('sample') or r.name}"
        sub = (f"File {Path(r.file).name}  ·  measured {info.get('timestamp', '?')}  ·  "
               f"{info.get('buffer', '')}" + (f" pH {info['bufferpH']}" if info.get("bufferpH") else "")
               + f"  ·  calibration: {r.calibration.calibrant or 'custom'} "
               f"({(r.calibration.source or {}).get('file', 'loaded')})")
        _header(fig, title, sub)
        if info.get("comments"):
            fig.text(0.04, 0.89, f"Comment: {info['comments']}", fontsize=8.5, color=INK2, va="top",
                     style="italic")
        gs = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.84, bottom=0.08,
                              width_ratios=[1.75, 1], wspace=0.08)

        # butterfly histogram: binding up, unbinding down
        ax = fig.add_subplot(gs[0, 0])
        bw = r.settings.bin_width
        xmax = _auto_plot_max(r)
        edges = np.arange(0, xmax + bw, bw)
        up, _ = np.histogram(r.events.mass[r.events.mass > 0], edges)
        down, _ = np.histogram(-r.events.mass[r.events.mass < 0], edges)
        ax.stairs(up, edges, fill=True, color=C_DATA, alpha=0.9, label="binding")
        ax.stairs(-down, edges, fill=True, color=C_UNBIND, alpha=0.9, label="unbinding (mirrored)")
        ax.axhline(0, color="#888888", lw=0.8)
        lo_lim = r.settings.mass_range[0]
        ax.axvspan(0, lo_lim, color="#000000", alpha=0.05, lw=0, label=f"below fit range (<{lo_lim:.0f} kDa)")

        if r.rois:
            _draw_rois(ax, r.rois, bw)
        else:
            xx = np.linspace(r.fit.edges[0], min(r.fit.edges[-1], xmax), 3000)
            for i, curve in enumerate(r.fit.peak_curves(xx)):  # fit and plot share the bin width
                ax.plot(xx, curve, color=C_FIT, lw=1, label="Gaussian fits" if i == 0 else None)
            if r.fit.background is not None:
                ax.plot(xx, gauss(xx, *r.fit.background), color=C_FIT, lw=0.8, ls=":")
        ytop = max(up.max() if len(up) else 1, 1)
        ax.set_ylim(-max(down.max() if len(down) else 1, ytop * 0.25) * 1.1, ytop * 1.28)

        # peak labels, staggered when close together
        last_x, level = -np.inf, 0
        for p in r.peaks.sort_values("mass_kDa").itertuples():
            level = level + 1 if p.mass_kDa - last_x < 0.09 * xmax else 0
            last_x = p.mass_kDa
            h = p.counts * bw / (p.sigma_kDa * np.sqrt(2 * np.pi))
            y = h + ytop * (0.05 + 0.13 * level)
            ax.annotate(f"{p.mass_kDa:.0f} kDa\n{p.percent:.0f}%", (p.mass_kDa, y), ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color=C_CHECK if p.notes else C_FIT,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        ax.set_xlim(0, xmax)
        ax.set(xlabel="Mass (kDa)", ylabel=f"Events per {bw:g} kDa",
               title="Mass distribution (binding ↑, unbinding ↓)"
                     + ("  ·  manual ROI fit" if r.rois else ""))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
        ax.legend(frameon=False, loc="upper right", fontsize=8)

        # right column: peaks table, QC, warnings (one axes, filled top-down)
        ax = fig.add_subplot(gs[0, 1])
        f0 = lambda v: f"{v:.0f}"
        y = _table(ax, r.peaks[["peak", "mass_kDa", "mass_err_kDa", "sigma_kDa", "counts", "percent"]],
                   ["#", "mass kDa", "± fit", "σ kDa", "events", "% *"],
                   [str, f0, lambda v: f"{v:.1f}" if np.isfinite(v) else "—", f0, f0, lambda v: f"{v:.1f}"],
                   title="Mass peaks",
                   widths=[0.08, 0.2, 0.15, 0.17, 0.2, 0.2], highlight=list(r.peaks["notes"].astype(bool)))
        q = r.qc
        y = _text_block(ax, [(f"* share of binding events in the fitted range "
                              f"({r.settings.mass_range[0]:.0f}–{r.settings.mass_range[1]:.0f} kDa). "
                              f"{q['assigned_to_peaks_pct']:.0f}% of these belong to a fitted peak; the rest "
                              f"is spread between peaks or below the lowest peak.", INK2)], wrap=62, size=7.5,
                        y0=y - 0.015)
        y = _text_block(ax, [
            (f"Events: {q['events']} (binding {q['binding_events']}, unbinding {q['unbinding_events']})", "#222222"),
            (f"Binding events in fitted range: {q['binding_in_mass_range']}", "#222222"),
            (f"Landing rate, first 10 s: {q['landing_rate_first_10s']:.0f} /s  ·  movie {q['duration_s']:.0f} s",
             "#222222"),
            (f"Calibration: mass = {r.calibration.slope:.1f} × contrast {r.calibration.intercept:+.2f} kDa "
             f"(valid {r.calibration.mass_range[0]:.0f}–{r.calibration.mass_range[1]:.0f} kDa)"
             if r.calibration.mass_range else
             f"Calibration: mass = {r.calibration.slope:.1f} × contrast {r.calibration.intercept:+.2f} kDa", INK2),
        ], title="Quality control", wrap=62, y0=y - 0.04)
        if r.rois:
            risk_col = {"low": "#2e7d32", "moderate": "#222222", "high": C_CHECK}
            y = _text_block(ax, [(f"[{x.risk} risk] {x.model_summary()}", risk_col[x.risk]) for x in r.rois],
                            title="Model check (BIC, per ROI)", wrap=62, y0=y - 0.03)
        warn = [(f"⚠ {w}", C_CHECK) for w in (r.calibration.warnings + r.warnings)] \
            or [("✓ No warnings", "#2e7d32")]
        if len(warn) > 8:
            warn = warn[:8] + [(f"… {len(warn) - 8} more in warnings.txt", INK2)]
        _text_block(ax, warn, title="Warnings", wrap=62, y0=y - 0.04)
        return fig


def _draw_rois(ax, rois, bw):
    """Shade ROIs and draw each mixture component (+ flat background) in events per bin."""
    first = True
    for roi in rois:
        ax.axvspan(roi.lo, roi.hi, color=C_CHECK, alpha=0.06, lw=0)
        ax.axvline(roi.lo, color=C_CHECK, lw=0.6, alpha=0.5)
        ax.axvline(roi.hi, color=C_CHECK, lw=0.6, alpha=0.5)
        xx = np.linspace(roi.lo, roi.hi, 800)
        comps, bg = roi.fit.density(xx, bw, per_component=True)
        for c in comps:
            ax.plot(xx, c, color=C_FIT, lw=1, label="Gaussian fits" if first else None)
            first = False
        if bg is not None:
            ax.plot(xx, bg, color=C_FIT, lw=0.8, ls=":", label="flat background" if roi is rois[0] else None)
        ax.plot(xx, roi.fit.density(xx, bw), color=C_FIT, lw=0.6, alpha=0.5)
        ax.text((roi.lo + roi.hi) / 2, 0.01, f"ROI: k={roi.k}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color=C_CHECK)


# ---------------------------------------------------------------------------- output
def write_reports(res, out) -> None:
    with plt.ioff():
        _write_reports(res, out)


def _write_reports(res, out) -> None:
    out = Path(out)
    figs = []
    fig = plot_calibration(res.calibration)
    fig.savefig(out / "calibration" / "calibration_report.pdf")
    fig.savefig(out / "calibration" / "calibration_report.png", dpi=130)
    figs.append(fig)
    for r in res.samples:
        fig = plot_sample(r)
        fig.savefig(out / r.name / "report.pdf")
        fig.savefig(out / r.name / "report.png", dpi=130)
        figs.append(fig)
    with PdfPages(out / "all_reports.pdf") as pdf:
        for f in figs:
            pdf.savefig(f)
    for f in figs:
        plt.close(f)
