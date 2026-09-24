# mprfile: read and analyse Refeyn mass photometry `.mpr` files in Python

> **Disclaimer:** this is an independent, unofficial project. It is not affiliated with, endorsed by or supported by Refeyn Ltd. "Refeyn", "AcquireMP", "DiscoverMP" and "MassFerence" are trademarks of their respective owners. The reader was written from scratch by inspecting output files with standard HDF5 tools, to make users' own measurement data accessible (interoperability). It contains no Refeyn code. Use at your own risk and check results against the official software.

From raw AcquireMP files to calibrated masses, peak tables and PDF reports, without DiscoverMP.

## Install (once)
You need [Miniforge](https://conda-forge.org/download/) or Anaconda. In the *Miniforge/Anaconda Prompt*, from this folder:
```
conda env create -f environment.yml
```
Every dependency has a lower and an upper version bound. The notebooks were tested at both ends of every range (oldest: numpy 1.26 / pandas 2.1 / h5py 3.10; newest: numpy 2.4 / pandas 3.0 / h5py 3.16). Without conda: `pip install -e ".[tiff,notebooks]"`.

## Get masses for your samples

Put **one calibrant measurement** (e.g. MassFerence P1, with *ladder* or *calib* in the file or sample name) and your **sample measurements** in one folder, measured on the same day with the same settings.

**No coding: notebook**
```
conda activate mpr
jupyter lab
```
Open `analyze_measurements.ipynb`, point `DATA_FOLDER` at your files, and choose **Run → Run All Cells**.

**No coding: one command**
```
conda activate mpr
cd C:\path\to\your\data
mpr-analyze
```
Both write a `results` folder:

| File | What it contains |
|---|---|
| `all_reports.pdf` | calibration page and one page per sample |
| `summary.csv` | one row per peak for all samples: mass, ± fit, σ, events, %, notes |
| `calibration/` | report, reusable `calibration.json`, residuals, ladder check |
| `<sample>/` | `report.pdf/.png`, `peaks.csv`, `events.csv` (every particle with its mass), `warnings.txt` |

Example: sample *252* against MassFerence P1 gives 552 kDa (37%) and 460 kDa (16%). It also reports a broad low-mass population (~72 kDa), marked as below the calibrated range and not a single clean peak.

### What the analysis does
1. **Calibration.** Peaks in the calibrant's contrast histogram are detected, fitted and matched automatically to the known masses (MassFerence P1: 86, 172, 258, 344 kDa). A straight line is then fitted through them. The higher ladder peaks (430, 516, 602 kDa) are not fitted; they are used as an independent check of linearity, which extends the validated range to 602 kDa.
2. **Samples.** AcquireMP's detected particles are converted to mass. Binding events are histogrammed, and peaks are detected and fitted as Gaussians.
3. **Quality control.** Warnings are raised for: calibrant peaks that are missing or have few events, poor linearity, stray unassigned peaks (a wrong calibrant), instrument or camera settings that differ between calibrant and sample, a calibrant from another day, peaks outside the calibrated range, peaks too broad to be one species, and low event counts.

## Expert mode
```
mpr-analyze data\*.mpr -c data\002_Ladder.mpr -o results          # explicit calibrant file
mpr-analyze new\*.mpr --calibration results\calibration\calibration.json   # reuse a calibration
mpr-analyze -k "66,132,480" ...                                     # other calibrant: masses in kDa
mpr-analyze --mass-range 30 2000 --bin-width 4 --background --max-fit-error 0.2
mpr-analyze --help
```
```python
from mprfile import calibrate, analyze, analyze_sample, AnalysisSettings, Calibration, MPRFile

cal = calibrate("002_Ladder.mpr", "MassFerence P1")       # or [86, 172, 258, 344]
print(cal.report()); cal.save("calibration.json")
r = analyze_sample("019_252_50nM.mpr", cal, AnalysisSettings(bin_width=4))
r.peaks, r.events, r.qc, r.warnings
res = analyze("data/*.mpr", calibrant_file="data/002_Ladder.mpr", out="results")
res.summary
```
Low-level access to everything in the file:
```python
with MPRFile("002_Ladder.mpr") as m:
    m.summary(); m.metadata()          # all settings, sample info, instrument
    m.frames(); m.ratiometric()        # raw and ratiometric movies
    m.events_df(calibration=cal)       # AcquireMP's fitted particles (+ mass)
    m.gaussian_fits(); m.scores()
    m.export_movie_tiff("movie.tif", ratiometric=True)   # for Fiji (needs tifffile)
```
`mpr_tutorial.ipynb` walks through the file contents and further analyses: the movies, your own peak fitting, ladder linearity, landing kinetics, drift and the average PSF.

### Adding a calibrant
Calibrants live in `mprfile/calibration.py` (`CALIBRANTS`). Add a name, its certified masses and, for an oligomer ladder, `unit_mass`. Or just pass the masses.

## File format
`.mpr` files are ordinary **HDF5** files (`movie/`, `analysis/`, `display/`). The only non-standard parts are:

| Part | What it is | How it's handled |
|---|---|---|
| `movie/frame` compression | HDF5 Zstandard filter (id 32015) + shuffle | `import hdf5plugin` |
| `movie/frame` encoding `TIME_DIFFERENCE` | frame 0 raw, then `frame[i]-frame[i-1]` mod 2¹⁶ | uint16 cumulative sum (checked against all stored keyframes) |
| `x__is_none__` / `__is_list__` datasets | how the software serialises Python `None` and lists | converted back by `metadata()` |

## Notes
- **Sign:** on this instrument (`scatter_interference_destructive = False`) a landing particle makes the raw pixels *brighter*. AcquireMP reports binding as **negative** contrast and positive mass, and `ratiometric()` does the same by default.
- **Ratiometric frames:** `ratiometric()[i]` = mean(F[i:i+n]) / mean(F[i−n:i]) − 1, so frame indices line up with `events()['frame']`. At the fitted positions it correlates with AcquireMP's contrasts at r ≈ 0.99.
- **Detected particles:** the analysis uses the particles AcquireMP detected and fitted; the calibration and peak fitting are done independently of AcquireMP.
- **Versions:** the format was reverse-engineered from AcquireMP 2025.1.2 (file format v4, movie format v20). Files from other versions may differ slightly.

## Data
No measurement data is included in this repository. `.gitignore` excludes `.mpr`, `.csv`, `.tif`, `results/` and exports, so that measurement data (which can include operator names and instrument serial numbers) isn't committed by accident.

## License
MIT, see [LICENSE](LICENSE).
