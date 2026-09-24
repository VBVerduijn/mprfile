"""
mprfile — read and analyse Refeyn mass photometry .mpr files.

Reading:     MPRFile            (raw/ratiometric movies, events, metadata)
Calibration: calibrate, Calibration, CALIBRANTS
Analysis:    analyze, analyze_sample, AnalysisSettings
Manual ROIs: mprfile.mixture (statistics), mprfile.interactive.ROIFitter (notebook tool)
Command:     mpr-analyze        (see `mpr-analyze --help`)
"""
__version__ = "0.3.0"

from .reader import MPRFile, load_events
from .calibration import CALIBRANTS, Calibration, calibrate
from .analysis import AnalysisSettings, analyze, analyze_sample, find_calibrant_file

__all__ = ["MPRFile", "load_events", "CALIBRANTS", "Calibration", "calibrate",
           "AnalysisSettings", "analyze", "analyze_sample", "find_calibrant_file", "__version__"]
