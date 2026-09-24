"""
Regenerate the wiki figures from *simulated* demo data (no real measurements involved).

    python tools/make_wiki_figures.py [output_folder]      # default: wiki/images

The interactive-tool screenshots (roi_fitter_*.png) are made separately in a browser.
"""
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mprfile import AnalysisSettings, MPRFile, analyze_sample, calibrate  # noqa: E402
from mprfile.demo import make_demo_files  # noqa: E402
from mprfile.mixture import fit_roi  # noqa: E402
from mprfile.report import plot_calibration, plot_sample  # noqa: E402

C_DATA, C_FIT, C_ACCENT = "#3b6ea5", "#444444", "#d9722e"
STYLE = {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
         "font.size": 9, "figure.dpi": 110}


def save(fig, out, name, dpi=110):
    fig.savefig(out / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / name)


def main(out="wiki/images"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    ladder, sample = make_demo_files(tmp)
    cal = calibrate(ladder, "MassFerence P1")

    # --- reports
    save(plot_calibration(cal), out, "calibration_report.png", dpi=90)
    auto = analyze_sample(sample, cal)
    save(plot_sample(auto), out, "sample_report.png", dpi=90)
    roi = analyze_sample(sample, cal, AnalysisSettings(rois=[(100, 200, 1), (230, 560, 2)]))
    save(plot_sample(roi), out, "sample_report_roi.png", dpi=90)

    with plt.rc_context(STYLE):
        # --- raw vs ratiometric frame
        with MPRFile(sample) as m:
            movie = m.frames()
            r = m.ratiometric()
            ev = m.events()
        counts = np.bincount(ev["frame"].astype(int), minlength=len(movie))
        f0 = int(np.nonzero(counts == int(np.percentile(counts[counts > 0], 90)))[0][0])   # a typical busy frame
        fig, axs = plt.subplots(1, 2, figsize=(11, 2.6))
        axs[0].imshow(movie[f0], cmap="gray")
        axs[0].set_title(f"Raw frame {f0}: particles are invisible")
        lim = np.nanpercentile(np.abs(r[f0]), 99.7)
        axs[1].imshow(r[f0], cmap="gray", vmin=-lim, vmax=lim)
        sel = ev["frame"].astype(int) == f0
        axs[1].scatter(ev["x"][sel], ev["y"][sel], s=90, facecolors="none", edgecolors=C_ACCENT, lw=1.3)
        axs[1].set_title("Ratiometric frame: landings appear as dark spots (fitted events circled)")
        for a in axs:
            a.axis("off")
        save(fig, out, "raw_vs_ratiometric.png")

        # --- one landing event over time
        k = np.nonzero((ev["contrast"] < -0.008) & (ev["nn_distance"] > 10))[0][0]
        fk, xk, yk = int(ev["frame"][k]), ev["x"][k], ev["y"][k]
        offs = [-6, -3, -1, 0, 1, 3, 6]
        fig, axs = plt.subplots(1, len(offs), figsize=(12, 2.0))
        for a, o in zip(axs, offs):
            a.imshow(r[fk + o], cmap="gray", vmin=-0.012, vmax=0.012)
            a.set_xlim(xk - 12, xk + 12); a.set_ylim(yk + 12, yk - 12)
            a.set_title("landing" if o == 0 else f"{o:+d} frames", fontsize=9); a.axis("off")
        save(fig, out, "landing_event.png")

        # --- contrast -> mass histogram of the demo sample (butterfly shown in reports)
        # --- k comparison in one ROI
        mass = auto.events.mass[auto.events.mass > 0].values
        lo, hi = 230, 560
        edges = np.arange(lo, hi + 5, 5)
        h, _ = np.histogram(mass, edges)
        base = fit_roi(mass, lo, hi, 2, resolution=cal.resolution_kDa)
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.1), sharey=True)
        xx = np.linspace(lo, hi, 700)
        for a, kk in zip(axs, [1, 2, 4]):
            rk = fit_roi(mass, lo, hi, kk, resolution=cal.resolution_kDa, models=base.models)
            a.stairs(h, edges, fill=True, color=C_DATA, alpha=0.8)
            comps, bg = rk.fit.density(xx, 5, per_component=True)
            for c in comps:
                a.plot(xx, c, color=C_FIT, lw=1)
            if bg is not None:
                a.plot(xx, bg, color=C_FIT, lw=0.8, ls=":")
            a.plot(xx, rk.fit.density(xx, 5), color=C_ACCENT, lw=1.5)
            d = rk.models.set_index("k").loc[kk, "dBIC"]
            a.set_title(f"k = {kk}  ·  ΔBIC = {d:+.1f}  ·  {rk.risk} risk", fontsize=9.5,
                        color=C_ACCENT if rk.risk == "high" else "#2e7d32")
            a.set_xlabel("Mass (kDa)")
        axs[0].set_ylabel("Events per 5 kDa")
        save(fig, out, "roi_k_comparison.png")

        # --- BIC / AIC vs k
        t = base.models
        cap = 30
        fig, ax = plt.subplots(figsize=(5.6, 3.1))
        ax.plot(t.k, np.minimum(t.dBIC, cap), "o-", color=C_DATA, lw=1.6, label="ΔBIC (used for the verdict)")
        ax.plot(t.k, np.minimum(t.dAIC, cap), "s--", color=C_FIT, mfc="white", lw=1, label="ΔAIC (for reference)")
        for kk, dv in zip(t.k, t.dBIC):
            if dv > cap:
                ax.annotate(f"{dv:.0f} ↑", (kk, cap), xytext=(6, -4), textcoords="offset points", fontsize=8,
                            color=C_DATA)
        for yv, lab in [(2, "positive"), (6, "strong"), (10, "very strong")]:
            ax.axhline(yv, color="#999", lw=0.7, ls=":")
            ax.text(4.35, yv, f"{lab} ≥{yv}", fontsize=7.5, color="#666", va="center")
        best = int(t.loc[t.dBIC.idxmin(), "k"])
        ax.annotate("BIC best", (best, 0), xytext=(28, 22), textcoords="offset points", ha="center", fontsize=8,
                    color="#2e7d32", arrowprops=dict(arrowstyle="->", color="#2e7d32"))
        ax.set(xlabel="Number of Gaussians k", ylabel="Δ vs best model", xticks=list(t.k), xlim=(0.7, 4.9),
               ylim=(-2, cap + 3), title=f"Model comparison, ROI {lo}–{hi} kDa")
        ax.legend(frameon=False, loc="upper center", fontsize=8)
        save(fig, out, "bic_vs_k.png")

        # --- Ashman's D illustration
        fig, axs = plt.subplots(1, 3, figsize=(11, 2.3), sharey=True)
        xx = np.linspace(-60, 100, 600)
        for a, sep in zip(axs, [15, 28, 45]):
            g1 = np.exp(-0.5 * (xx / 10) ** 2)
            g2 = np.exp(-0.5 * ((xx - sep) / 10) ** 2)
            dval = np.sqrt(2) * sep / np.sqrt(200)
            a.fill_between(xx, g1 + g2, color=C_DATA, alpha=0.25)
            a.plot(xx, g1, color=C_FIT, lw=1); a.plot(xx, g2, color=C_FIT, lw=1)
            a.set_title(f"D = {dval:.1f}" + ("  → not resolved" if dval < 2 else "  → resolved"),
                        color=C_ACCENT if dval < 2 else "#2e7d32", fontsize=9.5)
            a.set_yticks([]); a.set_xlabel("mass offset (kDa), σ = 10 kDa")
        save(fig, out, "ashman_d.png")

    print("done")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
