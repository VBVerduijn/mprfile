"""
Generate the wiki's API-Reference.md from the docstrings and signatures in mprfile.

    python tools/make_api_md.py wiki/API-Reference.md
"""
import dataclasses
import inspect
import sys
import textwrap

import mprfile
from mprfile import analysis, calibration, demo, interactive, mixture, peaks, reader, report

SECTIONS = [
    ("Reading files", reader, ["MPRFile", "load_events"]),
    ("Calibration", calibration, ["calibrate", "Calibration", "CALIBRANTS", "get_calibrant"]),
    ("Analysis", analysis, ["analyze", "analyze_sample", "AnalysisSettings", "SampleResult", "BatchResult",
                            "find_calibrant_file", "write_sample", "update_summary"]),
    ("Manual ROIs and model comparison", mixture, ["fit_roi", "fit_rois", "bootstrap_roi", "ROIResult",
                                                   "fit_mixture", "compare_models", "MixtureFit",
                                                   "evidence_label", "ashman_d"]),
    ("Interactive tool", interactive, ["ROIFitter"]),
    ("Automatic peak finding", peaks, ["fit_peaks", "PeakFit"]),
    ("Reports", report, ["plot_calibration", "plot_sample", "write_reports"]),
    ("Demo data", demo, ["make_demo_files", "write_demo_mpr"]),
]


def doc(obj):
    d = inspect.getdoc(obj) or ""
    return textwrap.dedent(d).strip()


def sig(name, obj):
    try:
        s = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return name
    s = s.replace("<class '", "").replace("'>", "").replace("'", "")
    return f"{name}{s}"


def block(text):
    return "```\n" + text + "\n```\n" if text else ""


def document(mod, name, out):
    obj = getattr(mod, name)
    qual = f"mprfile.{mod.__name__.split('.')[-1]}.{name}" if mod is not reader else f"mprfile.{name}"
    if inspect.isclass(obj):
        kind = "dataclass" if dataclasses.is_dataclass(obj) else "class"
        out.append(f"### `{name}` <sub>({kind})</sub>\n")
        out.append(f"`{qual}`\n\n")
        if dataclasses.is_dataclass(obj):
            fields = [f"{f.name}: {f.type}" + ("" if f.default is dataclasses.MISSING else f" = {f.default!r}")
                      for f in dataclasses.fields(obj)]
            out.append("```python\n" + f"{name}(\n    " + ",\n    ".join(fields) + "\n)\n```\n")
        else:
            out.append("\n```python\n" + sig(name, obj.__init__).replace("(self, ", "(").replace("(self)", "()") + "\n```\n")
        out.append(doc(obj) + "\n\n")
        members = []
        field_names = {f.name for f in dataclasses.fields(obj)} if dataclasses.is_dataclass(obj) else set()
        for mname, m in inspect.getmembers(obj):
            if mname.startswith("_") or mname in field_names:
                continue
            raw = inspect.getattr_static(obj, mname)
            if isinstance(raw, property):
                members.append((mname, "property", None, doc(raw)))
            elif isinstance(raw, (classmethod, staticmethod)) or inspect.isfunction(raw):
                f = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
                s = sig(mname, f).replace("(self, ", "(").replace("(self)", "()").replace("(cls, ", "(")
                members.append((mname, "classmethod" if isinstance(raw, classmethod) else "method", s, doc(f)))
        if members:
            out.append("\n| Member | Description |\n|---|---|\n")
            for mname, kind, s, d in members:
                first = d.split("\n\n")[0].replace("\n", " ").replace("|", "\\|") if d else ""
                label = (f"`{s}`" if s else f"`.{mname}` *(property)*").replace("|", "\\|")
                out.append(f"| {label} | {first} |\n")
            out.append("\n")
    elif callable(obj):
        out.append(f"### `{name}()`\n")
        out.append("```python\n" + sig(qual, obj) + "\n```\n")
        out.append(doc(obj) + "\n")
    else:
        out.append(f"### `{name}`\n")
        out.append(f"`{qual}` — constant.\n")
        if isinstance(obj, dict):
            out.append("```python\n" + "\n".join(f"{k!r}: masses {v.get('masses')} kDa  ({v.get('source', '')})"
                                                  for k, v in obj.items()) + "\n```\n")
    out.append("\n")


def main(path="wiki/API-Reference.md"):
    out = [f"# API reference\n\n*Generated from the docstrings of mprfile {mprfile.__version__} by "
           "`tools/make_api_md.py`. Everything listed under \"Reading files\", \"Calibration\" and "
           "\"Analysis\" can be imported directly: `from mprfile import MPRFile, calibrate, analyze, ...`.*\n\n"]
    out.append("**Contents**\n\n")
    for title, mod, names in SECTIONS:
        out.append(f"- **{title}** (`{mod.__name__}`): " + ", ".join(f"`{n}`" for n in names) + "\n")
    out.append("\n---\n\n")
    for title, mod, names in SECTIONS:
        out.append(f"## {title}\n\n")
        md = doc(mod).split("\n\n")[0]
        if md:
            out.append(f"*Module `{mod.__name__}`:* " + md.split("—", 1)[-1].strip() + "\n\n")
        for n in names:
            document(mod, n, out)
        out.append("---\n\n")
    out.append("## Command line\n\n`mpr-analyze` (module `mprfile.cli`). Full option list:\n\n")
    import io
    from contextlib import redirect_stdout
    from mprfile.cli import _parser
    buf = io.StringIO()
    with redirect_stdout(buf):
        _parser().print_help()
    out.append(block(buf.getvalue().rstrip()))
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main(*sys.argv[1:2])
