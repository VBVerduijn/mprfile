# mprfile: read Refeyn mass photometry `.mpr` files in Python

> **Disclaimer:** this is an independent, unofficial project. It is not affiliated with, endorsed by or supported by Refeyn Ltd. "Refeyn", "AcquireMP" and "DiscoverMP" are trademarks of their respective owners. The reader was written from scratch by inspecting output files with standard HDF5 tools, to make users' own measurement data accessible (interoperability). It contains no Refeyn code. Use at your own risk and check results against the official software.

`.mpr` files from Refeyn AcquireMP are ordinary **HDF5** files. The only non-standard parts are:

| Part | What it is | How it's handled |
|---|---|---|
| `movie/frame` compression | HDF5 Zstandard filter (id 32015) + shuffle | `import hdf5plugin` |
| `movie/frame` encoding `TIME_DIFFERENCE` | frame 0 raw, then `frame[i]-frame[i-1]` mod 2¹⁶ | uint16 cumulative sum (checked against all stored keyframes) |
| `x__is_none__` / `__is_list__` datasets | how the software serialises Python `None` and lists | converted back by `metadata()` |

## Install
With conda (recommended, for colleagues too). Run this from this folder:
```
conda env create -f environment.yml
conda activate mpr
jupyter lab                      # open mpr_tutorial.ipynb
```
The env installs `mprfile` in editable mode, so `import mprfile` works from any folder and edits to `mprfile.py` take effect right away. Every dependency has a lower and an upper version bound. The notebook was tested at both ends of every range (oldest: numpy 1.26 / pandas 2.1 / h5py 3.10; newest: numpy 2.4 / pandas 3.0 / h5py 3.16).

Without conda: `pip install -e ".[extras]"`

## Use
```python
from mprfile import MPRFile, Calibration

with MPRFile("002_Ladder.mpr") as m:
    print(m.summary())
    m.metadata()              # full settings/sample tree as a dict
    m.sample_info, m.camera, m.instrument, m.analysis_params

    movie = m.frames()        # (n, 59, 150) uint16 raw movie
    f = m.frame(1234)         # random access via keyframes
    r = m.ratiometric()       # ratiometric contrast movie (AcquireMP sign convention)
    t = m.times()             # frame times in s

    ev = m.events()           # AcquireMP's fitted events: contrast, x, y, frame, time, errors...
    df = m.events_df()        # same as a pandas DataFrame
    m.gaussian_fits()         # peaks fitted in the GUI histogram
    m.scores()                # per-frame brightness/sharpness/saturation

    cal = Calibration.fit(peak_contrasts, known_masses_kDa)
    ev = m.events(calibration=cal)          # adds ev["mass"]
    m.export_events_csv("events.csv", calibration=cal)
    m.export_movie_tiff("movie.tif", ratiometric=True)   # for Fiji
```

## Data
No measurement data is included in this repository. Use your own `.mpr` files. In the tutorial notebook, set `DATA = Path("your_file.mpr")` in the first code cell. `.gitignore` excludes `.mpr`, `.csv`, `.tif` and export files so that measurement data (which can include operator names and instrument serial numbers) isn't committed by accident.

## Tutorial
`mpr_tutorial.ipynb` covers opening a file, the metadata, the raw and ratiometric movies, the events and the histogram, and further analyses: your own peak fitting, ladder linearity and calibration, landing kinetics, drift, the average PSF, batch-processing and exports.

## Notes
- Sign: on this instrument (`scatter_interference_destructive = False`) a landing particle makes the raw pixels *brighter*. AcquireMP reports binding as **negative** contrast, and `ratiometric()` does the same by default (`acquiremp_sign=False` gives the raw sign).
- `ratiometric()[i]` = mean(F[i:i+n]) / mean(F[i−n:i]) − 1, so frame indices line up with `events()['frame']`. At the fitted positions it correlates with AcquireMP's contrasts at r ≈ 0.99. Single-pixel values are about 20% lower than the PSF-fit amplitudes, as expected.
- This file has no mass calibration stored (`calibrated_values` are NaN), so masses need a `Calibration`.
- The format was reverse-engineered from AcquireMP 2025.1.2 (file format v4, movie format v20). Files from other versions may differ slightly.

## License
MIT, see [LICENSE](LICENSE).
