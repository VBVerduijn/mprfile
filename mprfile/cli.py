"""
mpr-analyze — calibrate and analyse Refeyn .mpr files from the command line.

Simplest use (in a folder with one calibrant file and the sample files):
    mpr-analyze

Typical use:
    mpr-analyze samples/*.mpr --calibrant-file 002_Ladder.mpr --out results

Run `mpr-analyze --help` for all options.
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .analysis import AnalysisSettings, analyze
from .calibration import CALIBRANTS


def _parser() -> argparse.ArgumentParser:
    d = AnalysisSettings()
    p = argparse.ArgumentParser(
        prog="mpr-analyze",
        description="Calibrate with a calibrant measurement and determine masses in Refeyn .mpr "
                    "sample files. Writes per-sample reports (PDF/PNG), peak tables and event "
                    "lists (CSV) and a summary.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("samples", nargs="*", default=["*.mpr"],
                   help="sample files, folders or wildcards")
    g = p.add_argument_group("calibration")
    g.add_argument("-c", "--calibrant-file", help="calibrant .mpr file (default: auto-detect a file "
                   "whose name contains ladder/calib/massference/mpf/standard)")
    g.add_argument("-k", "--calibrant", default="MassFerence P1",
                   help=f"calibrant name ({', '.join(CALIBRANTS)}) or masses in kDa, e.g. '66,132,480'")
    g.add_argument("--calibration", help="reuse a saved calibration.json instead of a calibrant file")
    g = p.add_argument_group("output")
    g.add_argument("-o", "--out", default="results", help="output folder")
    g.add_argument("--no-reports", action="store_true", help="skip PDF/PNG reports (CSV only)")
    g.add_argument("-q", "--quiet", action="store_true")
    g = p.add_argument_group("expert settings")
    g.add_argument("--mass-range", nargs=2, type=float, metavar=("MIN", "MAX"), default=d.mass_range,
                   help="kDa window for peak fitting")
    g.add_argument("--bin-width", type=float, default=d.bin_width, help="histogram bin width (kDa)")
    g.add_argument("--min-fraction", type=float, default=d.min_fraction,
                   help="ignore peaks with a smaller share of events")
    g.add_argument("--min-counts", type=int, default=d.min_counts, help="ignore peaks with fewer events")
    g.add_argument("--min-prominence", type=float, default=d.min_prominence,
                   help="peak detection threshold (fraction of highest bin)")
    g.add_argument("--sigma-max", type=float, default=d.sigma_max, help="max peak width σ (kDa)")
    g.add_argument("--max-fit-error", type=float, default=None,
                   help="drop events with AcquireMP PSF fit error above this")
    g.add_argument("--background", action="store_true", help="fit a broad background under the peaks")
    g.add_argument("--plot-max", type=float, default=None, help="right edge of the histogram (kDa)")
    p.add_argument("--version", action="version", version=f"mprfile {__version__}")
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    st = AnalysisSettings(mass_range=tuple(a.mass_range), bin_width=a.bin_width,
                          min_fraction=a.min_fraction, min_counts=a.min_counts,
                          min_prominence=a.min_prominence, sigma_max=a.sigma_max,
                          max_fit_error=a.max_fit_error, background=a.background, plot_max=a.plot_max)
    try:
        analyze(a.samples, calibrant_file=a.calibrant_file, calibrant=a.calibrant,
                calibration=a.calibration, out=a.out, settings=st, reports=not a.no_reports,
                verbose=not a.quiet)
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
