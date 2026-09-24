"""
mprfile.interactive — choose regions (ROIs) and the number of Gaussians by hand, with
live statistics that warn about overfitting.

In a Jupyter notebook:
    %matplotlib widget                  # enables dragging on the plot (ipympl)
    from mprfile.interactive import ROIFitter
    fitter = ROIFitter(results)         # results from analyze(), or a SampleResult
    fitter                              # shows the tool

Drag horizontally on the histogram to add an ROI (or type its limits), choose the number
of Gaussians per ROI, and read the model comparison: every ROI is also fitted with
1..k_max Gaussians and compared by BIC; overfitting and unreliable components are flagged.
"Save" writes the ROI analysis like the automatic one (report, peaks.csv, rois.json with
the exact ROI definition so the result can be reproduced).
"""
from __future__ import annotations

import copy
import html
from dataclasses import replace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .analysis import AnalysisSettings, SampleResult, analyze_sample, roi_peaks_table, update_summary, write_sample
from .mixture import bootstrap_roi, evidence_label, fit_roi

__all__ = ["ROIFitter"]

C_DATA, C_FIT, C_ROI, C_OK, C_WARN = "#3b6ea5", "#333333", "#d9722e", "#2e7d32", "#d9722e"
RISK_STYLE = {"low": (C_OK, "low risk"), "moderate": ("#8a6d00", "check notes"), "high": (C_WARN, "HIGH RISK")}


def _interactive_backend() -> bool:
    b = matplotlib.get_backend().lower()
    return "ipympl" in b or "widget" in b


class ROIFitter:
    """Interactive ROI / Gaussian-count selection with overfitting statistics.

    source:   a BatchResult (dropdown over its samples) or a single SampleResult.
    k_max:    largest number of Gaussians in the model comparison.
    background: flat background term in each ROI (recommended).
    """

    def __init__(self, source, *, k_max: int = 4, background: bool = True, bin_width: float | None = None,
                 plot_max: float | None = None, out=None):
        import ipywidgets as W  # noqa: F401  (fail early with a clear error if missing)
        if isinstance(source, SampleResult):
            self.samples, self.out = [source], out
        else:
            self.samples, self.out = list(source.samples), out or getattr(source, "out", None)
        if not self.samples:
            raise ValueError("no samples to show")
        self.k_max, self.background = k_max, background
        self._bin_width, self._plot_max = bin_width, plot_max
        self.state = {i: [] for i in range(len(self.samples))}    # per sample: list of ROIResult
        self.idx = 0
        self._build()
        # start from ROIs already used for a sample (e.g. from settings)
        for i, s in enumerate(self.samples):
            if s.rois:
                self.state[i] = list(s.rois)
        self._refresh()

    # ------------------------------------------------------------------ data helpers
    @property
    def sample(self) -> SampleResult:
        """The currently shown sample."""
        return self.samples[self.idx]

    @property
    def rois(self):
        """ROIs (ROIResult objects) of the currently shown sample."""
        return self.state[self.idx]

    def _masses(self):
        m = self.sample.events.mass.values
        return m[m > 0]

    @property
    def resolution(self):
        """Single-species width σ (kDa) from the calibration, used by the checks."""
        return self.sample.calibration.resolution_kDa

    @property
    def bin_width(self):
        """Histogram bin width used for display (kDa)."""
        return self._bin_width or self.sample.settings.bin_width

    # ------------------------------------------------------------------ public API
    def add_roi(self, lo: float, hi: float, k: int | None = None):
        """Add an ROI; k=None uses the BIC-preferred number of Gaussians."""
        lo, hi = sorted((float(lo), float(hi)))
        if hi - lo < 3 * self.bin_width:
            self._msg(f"ROI {lo:.0f}–{hi:.0f} kDa is too narrow.", error=True)
            return None
        self._busy(f"Fitting ROI {lo:.0f}–{hi:.0f} kDa with 1…{self.k_max} Gaussians…")
        try:
            r = fit_roi(self._masses(), lo, hi, k or 1, resolution=self.resolution, k_max=self.k_max,
                        background=self.background)
            if k is None and r.bic_best_k != r.k:
                r = fit_roi(self._masses(), lo, hi, r.bic_best_k, resolution=self.resolution,
                            k_max=self.k_max, background=self.background, models=r.models)
        except ValueError as exc:
            self._msg(str(exc), error=True)
            return None
        self.rois.append(r)
        self.rois.sort(key=lambda x: x.lo)
        self._refresh()
        return r

    def set_k(self, i: int, k: int):
        """Change the number of Gaussians of ROI i and refit (reuses the model comparison)."""
        r = self.rois[i]
        if k == r.k:
            return
        self._busy(f"Refitting ROI {r.label} with {k} Gaussian(s)…")
        try:
            self.rois[i] = fit_roi(self._masses(), r.lo, r.hi, int(k), resolution=self.resolution,
                                   k_max=self.k_max, background=r.background, models=r.models)
        except ValueError as exc:
            self._msg(str(exc), error=True)
        self._refresh()

    def remove_roi(self, i: int):
        """Delete ROI number i (0-based, ROIs are sorted by mass)."""
        del self.rois[i]
        self._refresh()

    def bootstrap(self, n: int = 100):
        """Bootstrap stability check for every ROI of the current sample."""
        for r in self.rois:
            self._busy(f"Bootstrap {n}× for ROI {r.label}…")
            bootstrap_roi(self._masses(), r, n, resolution=self.resolution)
        self._refresh()

    def settings(self) -> AnalysisSettings:
        """AnalysisSettings reproducing the current ROI choice for this sample."""
        return replace(self.sample.settings, rois=[r.spec() for r in self.rois],
                       roi_background=self.background, roi_k_max=self.k_max)

    def result(self) -> SampleResult:
        """The current sample re-analysed with the chosen ROIs (reuses the fits)."""
        s = copy.copy(self.sample)
        st = self.settings()
        n_range = s.qc["binding_in_mass_range"]
        s.rois = list(self.rois)
        s.settings = st
        s.peaks = roi_peaks_table(s.rois, n_range, s.calibration)
        s.qc = dict(s.qc, assigned_to_peaks_pct=round(float(s.peaks.percent.sum()), 1) if len(s.peaks) else 0.0)
        base = [w for w in s.warnings if not w.startswith(("ROI ", "Peak "))]
        s.warnings = base + [f"ROI {r.label}: {w}" for r in s.rois for w in r.warnings]
        return s

    def save(self, out=None):
        """Write the ROI analysis of the current sample (report, CSVs, rois.json)."""
        out = out or self.out or "results"
        if not self.rois:
            self._msg("Add at least one ROI before saving.", error=True)
            return None
        r = self.result()
        d = write_sample(r, out, reports=True)
        update_summary(r, out)
        self._msg(f"Saved to {d}  (report.pdf, peaks.csv, rois.json, roi_model_comparison.csv); "
                  f"summary.csv updated.")
        return d

    # ------------------------------------------------------------------ UI
    def _build(self):
        import ipywidgets as W
        self.W = W
        self.live = _interactive_backend()
        with plt.ioff():
            self.fig, (self.ax, self.axr) = plt.subplots(
                2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [4, 1.1], "hspace": 0.08})
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.11)
        if self.live:
            from matplotlib.widgets import SpanSelector
            self.fig.canvas.header_visible = False
            self.fig.canvas.toolbar_position = "right"
            self.span = SpanSelector(self.ax, self._on_span, "horizontal", useblit=True, minspan=5,
                                     props=dict(alpha=0.2, facecolor=C_ROI), interactive=False)
            self.plot_box = self.fig.canvas
        else:
            self.plot_box = W.Output()

        sample_opts = [(f"{s.info.get('sample')} ({s.name})", i) for i, s in enumerate(self.samples)]
        self.w_sample = W.Dropdown(options=sample_opts, value=0, description="Sample",
                                   layout=W.Layout(width="360px"))
        self.w_sample.observe(lambda ch: self._switch(ch["new"]), names="value")
        self.w_lo = W.FloatText(description="from", layout=W.Layout(width="150px"))
        self.w_hi = W.FloatText(description="to", layout=W.Layout(width="150px"))
        self.w_k = W.Dropdown(options=[("BIC best", 0)] + [(str(k), k) for k in range(1, self.k_max + 1)],
                              value=0, description="Gaussians", layout=W.Layout(width="190px"))
        b_add = W.Button(description="Add ROI", icon="plus", button_style="primary")
        b_add.on_click(lambda _: self._add_from_fields())
        b_boot = W.Button(description="Bootstrap check (100×)", icon="refresh", layout=W.Layout(width="210px"),
                          tooltip="Refit 100 resampled data sets: are the components stable?")
        b_boot.on_click(lambda _: self.bootstrap(100))
        b_save = W.Button(description="Save", icon="save", button_style="success")
        b_save.on_click(lambda _: self.save())
        b_clear = W.Button(description="Clear ROIs", icon="trash")
        b_clear.on_click(lambda _: (self.rois.clear(), self._refresh()))

        hint = ("<b>Drag across the histogram</b> to add an ROI, or type its limits:" if self.live else
                "Type ROI limits and click <i>Add ROI</i>. (To draw ROIs with the mouse, run "
                "<code>%matplotlib widget</code> before creating the fitter.)")
        self.status = W.HTML()
        self.roi_box = W.VBox()
        self.stats = W.HTML()
        self.repro = W.HTML()
        self.ui = W.VBox([
            W.HBox([self.w_sample]) if len(self.samples) > 1 else W.HTML(""),
            self.plot_box,
            W.HTML(hint),
            W.HBox([self.w_lo, self.w_hi, self.w_k, b_add]),
            self.roi_box,
            W.HBox([b_boot, b_clear, b_save]),
            self.status, self.stats, self.repro,
        ])

    def _ipython_display_(self):
        from IPython.display import display
        display(self.ui)

    def show(self):
        """Display the tool (same as leaving `fitter` as the last line of a cell)."""
        from IPython.display import display
        display(self.ui)

    def _on_span(self, lo, hi):
        lo, hi = round(lo), round(hi)
        self.w_lo.value, self.w_hi.value = lo, hi
        self.add_roi(lo, hi, self.w_k.value or None)

    def _add_from_fields(self):
        self.add_roi(self.w_lo.value, self.w_hi.value, self.w_k.value or None)

    def _switch(self, i):
        self.idx = i
        self._refresh()

    def _busy(self, text):
        self.status.value = f"<span style='color:#555'>⏳ {html.escape(text)}</span>"

    def _msg(self, text, error=False):
        col = C_WARN if error else C_OK
        self.status.value = f"<span style='color:{col}'>{html.escape(text)}</span>"

    # ------------------------------------------------------------------ rendering
    def _refresh(self):
        self._draw()
        self._roi_rows()
        self._stats_html()
        specs = [r.spec() for r in self.rois]
        self.repro.value = ("<div style='color:#555;font-size:90%'>Reproduce this analysis: "
                            f"<code>AnalysisSettings(rois={specs!r})</code> or "
                            "<code>mpr-analyze " + " ".join(f"--roi {lo:g} {hi:g} {k}" for lo, hi, k in specs)
                            + "</code></div>") if specs else ""
        if not self.status.value.startswith("<span style='color:" + C_WARN):
            self.status.value = ""

    def _draw(self):
        ax, axr, bw = self.ax, self.axr, self.bin_width
        ax.clear(); axr.clear()
        m = self._masses()
        xmax = self._plot_max or self.sample.settings.plot_max or float(
            min(max(np.ceil(np.percentile(m, 99.5) * 1.15 / 100) * 100, 200), 2000))
        edges = np.arange(0, xmax + bw, bw)
        h, _ = np.histogram(m, edges)
        c = 0.5 * (edges[1:] + edges[:-1])
        ax.stairs(h, edges, fill=True, color=C_DATA, alpha=0.85, label="binding events")
        model = np.full_like(c, np.nan, dtype=float)
        for j, r in enumerate(self.rois):
            ax.axvspan(r.lo, r.hi, color=C_ROI, alpha=0.08, lw=0)
            xx = np.linspace(r.lo, r.hi, 600)
            comps, bgc = r.fit.density(xx, bw, per_component=True)
            for comp in comps:
                ax.plot(xx, comp, color=C_FIT, lw=1)
            if bgc is not None:
                ax.plot(xx, bgc, color=C_FIT, lw=0.8, ls=":")
            ax.plot(xx, r.fit.density(xx, bw), color=C_ROI, lw=1.4)
            col, tag = RISK_STYLE[r.risk]
            ax.text((r.lo + r.hi) / 2, 0.97, f"ROI {j + 1}: k={r.k}\n{tag}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color=col, fontweight="bold")
            for mu, s, w in zip(r.fit.mu, r.fit.sigma, r.fit.weights_in_roi):
                ax.annotate(f"{mu:.0f}", (mu, 0), xytext=(0, -2), textcoords="offset points",
                            ha="center", va="top", fontsize=7.5, color=C_FIT)
            inside = (c >= r.lo) & (c <= r.hi)
            model[inside] = r.fit.density(c[inside], bw)
        # residuals (Pearson): (data - model) / sqrt(model)
        with np.errstate(invalid="ignore", divide="ignore"):
            res = (h - model) / np.sqrt(np.where(model > 0, model, np.nan))
        axr.bar(c, np.nan_to_num(res), width=bw, color=np.where(np.abs(np.nan_to_num(res)) > 2, C_ROI, "#9aa5b1"))
        axr.axhline(0, color="#888", lw=0.8)
        for yv in (-2, 2):
            axr.axhline(yv, color="#bbb", lw=0.6, ls="--")
        axr.set_ylim(-4.5, 4.5)
        axr.set_ylabel("residual\n(σ)", fontsize=8)
        axr.set_xlabel("Mass (kDa)")
        ax.set_ylabel(f"Events per {bw:g} kDa")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, max(h.max(), 1) * 1.3)
        s = self.sample
        ax.set_title(f"{s.info.get('sample')} — {s.name}   (resolution σ ≈ "
                     f"{self.resolution:.0f} kDa from calibrant)" if self.resolution else s.name, fontsize=10)
        for a in (ax, axr):
            a.spines[["top", "right"]].set_visible(False)
            a.grid(alpha=0.25)
        if self.live:
            self.fig.canvas.draw_idle()
        else:
            self.plot_box.clear_output(wait=True)
            with self.plot_box:
                from IPython.display import display
                display(self.fig)

    def _roi_rows(self):
        W = self.W
        rows = []
        for i, r in enumerate(self.rois):
            col, tag = RISK_STYLE[r.risk]
            k_dd = W.Dropdown(options=list(range(1, self.k_max + 1)), value=r.k, description="Gaussians",
                              layout=W.Layout(width="170px"))
            k_dd.observe(lambda ch, i=i: self.set_k(i, ch["new"]), names="value")
            b_best = W.Button(description=f"use BIC best (k={r.bic_best_k})", disabled=(r.bic_best_k == r.k),
                              layout=W.Layout(width="190px"))
            b_best.on_click(lambda _, i=i, k=r.bic_best_k: self.set_k(i, k))
            b_del = W.Button(icon="times", tooltip="remove ROI", layout=W.Layout(width="40px"))
            b_del.on_click(lambda _, i=i: self.remove_roi(i))
            label = W.HTML(f"<b>ROI {i + 1}</b>: {r.label} &nbsp; <span style='color:{col};font-weight:bold'>"
                           f"{tag}</span>", layout=W.Layout(width="330px"))
            rows.append(W.HBox([label, k_dd, b_best, b_del]))
        self.roi_box.children = rows

    def _stats_html(self):
        if not self.rois:
            self.stats.value = ("<p style='color:#555'>No ROIs yet. Tip: start with the BIC-preferred "
                                "number of Gaussians and only deviate with a physical reason.</p>")
            return
        n_range = self.sample.qc["binding_in_mass_range"]
        parts = []
        for i, r in enumerate(self.rois):
            col, tag = RISK_STYLE[r.risk]
            t = r.models
            head = ("<tr><th>Gaussians</th><th>log L</th><th>params</th><th>AIC</th><th>BIC</th>"
                    "<th>ΔBIC</th><th>evidence against</th></tr>")
            body = ""
            for row in t.itertuples():
                chosen = row.k == r.k
                best = row.k == r.bic_best_k
                style = "font-weight:bold;background:#fdf1e8" if chosen else ""
                mark = (" ◀ chosen" if chosen else "") + (" ★ BIC best" if best else "")
                body += (f"<tr style='{style}'><td>{row.k}{mark}</td><td>{row.log_likelihood:.1f}</td>"
                         f"<td>{row.n_params}</td><td>{row.AIC:.1f}</td><td>{row.BIC:.1f}</td>"
                         f"<td>{row.dBIC:+.1f}</td><td>{row.evidence_against}</td></tr>")
            f = r.fit
            comp = "".join(
                f"<tr><td>{mu:.1f}</td><td>{'' if f.mu_err is None else (f'± {e:.1f}' if np.isfinite(e) else '—')}</td><td>{s:.1f}</td>"
                f"<td>{cnt:.0f}</td><td>{100 * cnt / max(n_range, 1):.1f}%</td>"
                + (f"<td>± {r.bootstrap.mass_sd.values[j]:.1f}</td>" if r.bootstrap is not None else "")
                + "</tr>"
                for j, (mu, e, s, cnt) in enumerate(zip(f.mu, f.mu_err if f.mu_err is not None else [0] * f.k,
                                                        f.sigma, f.counts)))
            bh = "<th>bootstrap SD</th>" if r.bootstrap is not None else ""
            warn = "".join(f"<li>{html.escape(w)}</li>" for w in r.warnings) or "<li>none</li>"
            bgtxt = f"flat background: {100 * f.background_share:.0f}% of the ROI's {f.n} events" \
                if f.has_background else f"{f.n} events, no background term"
            parts.append(f"""
<div style="border-left:4px solid {col};padding:4px 10px;margin:8px 0">
<b>ROI {i + 1}: {r.label}</b> — {r.k} Gaussian(s); {html.escape(r.model_summary().split('; ', 1)[1])}
<span style="color:{col};font-weight:bold">[{tag}]</span><br>
<small style="color:#555">{bgtxt}</small>
<table style="font-size:90%;margin-top:4px;border-collapse:collapse" cellpadding="3"><tr><th>mass (kDa)</th><th>± fit</th><th>σ (kDa)</th>
<th>events</th><th>% of range</th>{bh}</tr>{comp}</table>
<details><summary style="cursor:pointer;color:#3b6ea5">Model comparison (1…{self.k_max} Gaussians)</summary>
<table style="font-size:90%;border-collapse:collapse" cellpadding="3">{head}{body}</table>
<small style="color:#555">ΔBIC = BIC − lowest BIC. Evidence against a model: &lt;2 weak, 2–6 positive,
6–10 strong, &gt;10 very strong (Kass &amp; Raftery 1995). AIC is shown for reference; it favours
more components than BIC.</small></details>
<div style="color:{col if r.warnings else C_OK}"><b>Checks:</b><ul style="margin:2px 0">{warn}</ul></div>
</div>""")
        style = ("<style>.mpr-roi td,.mpr-roi th{padding:2px 10px;text-align:right}"
                 ".mpr-roi td:first-child,.mpr-roi th:first-child{text-align:left}</style>")
        self.stats.value = style + "<div class='mpr-roi'>" + "".join(parts) + "</div>"
