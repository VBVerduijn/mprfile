"""Example: inspect a Refeyn .mpr file with mprfile.py.

    python example.py 002_Ladder.mpr
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from mprfile import MPRFile, Calibration

path = sys.argv[1] if len(sys.argv) > 1 else "002_Ladder.mpr"

with MPRFile(path) as m:
    print(m.summary(), "\n")
    print("Movie decodes consistently with keyframes:", m.verify())

    ev = m.events()                       # dict of numpy arrays
    peaks = m.gaussian_fits()             # peaks fitted in AcquireMP's GUI
    print("\nGaussian fits stored in the file (contrast):")
    for p in peaks:
        print(f"  {p['position']:+.5f}  σ={p['sigma']:.5f}  counts={p['counts']:.0f}")

    # --- calibration example -------------------------------------------------
    # Pair binding-peak contrasts with the known masses of your standard, e.g.
    #   cal = Calibration.fit([-0.0112, -0.0084, -0.0056], [198, 132, 66])
    #   ev = m.events(calibration=cal)      # adds ev['mass']
    # (the masses above are placeholders; use your ladder's real values)

    r = m.ratiometric()                   # same frame indexing as ev['frame']
    busiest = int(np.bincount(ev["frame"].astype(int)).argmax())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.3, 1]})

    counts, edges = m.histogram(bin_width=2e-4, events=ev, range=(-0.04, 0.03))
    ax1.stairs(counts, edges, fill=True, color="#4a6fa5", alpha=0.85)
    for p in peaks:
        ax1.axvline(p["position"], color="0.35", lw=0.8, ls=":")
    ax1.set(xlabel="Ratiometric contrast", ylabel="Counts",
            title=f"{m.sample_info['sample']}: {len(ev['contrast'])} events")
    ax1.spines[["top", "right"]].set_visible(False)

    lim = np.nanpercentile(np.abs(r[busiest]), 99.5)
    im = ax2.imshow(r[busiest], cmap="gray", vmin=-lim, vmax=lim)
    sel = ev["frame"].astype(int) == busiest
    ax2.scatter(ev["x"][sel], ev["y"][sel], s=80, facecolors="none", edgecolors="#e07b39", lw=1.2)
    ax2.set_title(f"Ratiometric frame {busiest}: fitted events circled")
    ax2.axis("off")
    fig.colorbar(im, ax=ax2, fraction=0.025, label="contrast")

    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + "_overview.png"
    fig.savefig(out, dpi=150)
    print("\nSaved", out)
