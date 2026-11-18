import re
import io
import base64
from collections import defaultdict

_SPINNER_HTML = """
<style>@keyframes _spin{to{transform:rotate(360deg)}}</style>
<div style='font-family:monospace;padding:10px;display:flex;align-items:center;gap:10px;color:#555'>
  <div style='width:16px;height:16px;border-radius:50%;border:2px solid #ddd;
              border-top-color:#555;animation:_spin 0.7s linear infinite;flex-shrink:0'></div>
  Loading...
</div>"""

import pandas as pd
import glob
import os
from . import io_utils
from matplotlib.figure import Figure

_EXT_MAP = {
    "hst":      ".hst",
    "hdf5":     ".phdf",
    "rst":      ".rhdf",
    "openpmd":  ".bp",
    "spectrum": ".spc",
    "histogram": ".hdf",
}


# ---------------------------------------------------------------------------
# Lazy accordion helpers
# ---------------------------------------------------------------------------

def _lazy_accordion(title, render_fn):
    """Accordion that calls render_fn() -> HTML string on first open."""
    from IPython.display import display, HTML
    import ipywidgets as widgets
    out = widgets.Output()
    acc = widgets.Accordion(children=[out], selected_index=None)
    acc.set_title(0, title)
    done = [False]
    def on_open(change, out=out, done=done):
        if change['new'] == 0 and not done[0]:
            done[0] = True
            with out:
                display(HTML(_SPINNER_HTML))
            html = render_fn()
            out.clear_output(wait=True)
            with out:
                display(HTML(html))
    acc.observe(on_open, names='selected_index')
    return acc


def _lazy_widget_accordion(title, display_fn):
    """Accordion that calls display_fn() (uses IPython.display internally) on first open."""
    from IPython.display import display, HTML
    import ipywidgets as widgets
    out = widgets.Output()
    acc = widgets.Accordion(children=[out], selected_index=None)
    acc.set_title(0, title)
    done = [False]
    def on_open(change, out=out, done=done):
        if change['new'] == 0 and not done[0]:
            done[0] = True
            with out:
                display(HTML(_SPINNER_HTML))
            out.clear_output(wait=True)
            with out:
                display_fn()
    acc.observe(on_open, names='selected_index')
    return acc

# Imported here (after the helpers/constants above) because output_types
# imports them back from this module.
from .output_types import (
    History,
    OutputsCollection,
    OutputSeries,
    SpectrumOutputType,
    HistogramOutputType,
    OpenPMDOutputType,
    HDF5OutputType,
    EnergyTransferOutputType,
)

# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

class SimulationConfig(dict):
    """Wrapper around the parsed config dict with a rich Jupyter HTML repr."""

    def _repr_html_(self):
        style = """
        <style>
          .sim-config details { margin: 4px 0; }
          .sim-config summary {
            font-weight: bold; font-family: monospace;
            cursor: pointer; padding: 3px 6px;
            background: #f0f0f0; border-radius: 4px;
          }
          .sim-config table {
            border-collapse: collapse; margin: 4px 0 4px 16px;
            font-family: monospace; font-size: 0.9em;
          }
          .sim-config td { padding: 2px 12px; border-bottom: 1px solid #e0e0e0; }
          .sim-config td:first-child { color: #555; }
        </style>"""
        parts = [f'<div class="sim-config">{style}']
        for section, entries in self.items():
            rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in entries.items()
            )
            parts.append(
                f"<details><summary>{section}</summary>"
                f"<table>{rows}</table></details>"
            )
        parts.append("</div>")
        return "\n".join(parts)

    def __repr__(self):
        lines = []
        for section, entries in self.items():
            lines.append(f"[{section}]")
            for k, v in entries.items():
                lines.append(f"  {k} = {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class BaseSimulation:
    """
    Interface for analyzing Parthenon simulation data.

    Output types are handled by OutputType subclasses stored in
    self.output_types. Append a new handler to support additional file types:

        sim.output_types.append(MyOutputType())
    """

    _default_output_type_classes = [
        SpectrumOutputType,
        HistogramOutputType,
        OpenPMDOutputType,
        HDF5OutputType,
    ]

    def __init__(self, directory, pattern="parthenon.prim.*.phdf"):
        self.directory = directory
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")
        input_files = glob.glob(os.path.join(directory, "*.in"))
        if not input_files:
            raise FileNotFoundError(f"No input file (*.in) found in directory {directory}")
        if len(input_files) > 1:
            raise RuntimeError(f"Multiple input files found in directory {directory}: {input_files}")
        self.input_file = input_files[0]
        self.config = SimulationConfig(io_utils.parse_parthenon_input(self.input_file))
        self.output_types = [cls() for cls in self._default_output_type_classes]
        self._live_widgets = []

    def close(self):
        """Close ipywidgets created by a previous _ipython_display_ render.

        Each render builds a fresh tree of Accordion/Output widgets; if the old
        ones are never closed, their frontend comms stay registered and pile up
        across repeated re-displays in the same kernel session, which is how a
        notebook ends up rendering stale/duplicated widget output. Called
        automatically at the start of _ipython_display_, but can be called
        directly too (e.g. before deleting a Simulation you displayed).
        """
        def _close_recursive(w):
            for child in getattr(w, "children", ()):
                _close_recursive(child)
            try:
                w.close()
            except Exception:
                pass
        for w in self._live_widgets:
            _close_recursive(w)
        self._live_widgets = []

    def _output_info(self):
        handler_map = {h.file_type: h for h in self.output_types}
        outputs = []
        for section in self.config:
            m = re.match(r"parthenon/output(\d+)$", section)
            if not m:
                continue
            n = m.group(1)
            cfg = self.config[section]
            file_type = cfg.get("file_type", "unknown")
            dt = cfg.get("dt")
            if file_type == "hst":
                pattern = f"parthenon.out{n}{_EXT_MAP['hst']}"
                label = "hst"
            else:
                handler = handler_map.get(file_type)
                if handler is not None:
                    try:
                        pattern, label = handler.file_pattern(n, cfg)
                    except NotImplementedError:
                        continue
                else:
                    ext = _EXT_MAP.get(file_type, "")
                    oid = cfg.get("id", f"out{n}")
                    pattern = f"parthenon.{oid}.*{ext}"
                    label = f"{file_type} [{oid}]"
            files = sorted([
                f for f in glob.glob(os.path.join(self.directory, pattern))
                if not f.endswith(".xdmf")
            ])
            file_info = []
            for f in files:
                try:
                    idx = io_utils.extract_index(f)
                    time = idx * dt if dt is not None else None
                except ValueError:
                    idx, time = None, None
                file_info.append((idx, time, os.path.basename(f)))
            outputs.append((f"output{n}", label, file_type, file_info))
        return outputs

    @property
    def outputs(self):
        """OutputsCollection mapping output id → OutputSeries for all non-history outputs."""
        handler_map = {h.file_type: h for h in self.output_types}
        result = OutputsCollection()
        for name, label, file_type, files in self._output_info():
            if file_type == "hst":
                continue
            handler = handler_map.get(file_type)
            key = label.split("[")[-1].rstrip("]") if "[" in label else name
            result[key] = OutputSeries(name, label, files, handler, self.directory)
        return result

    def energy_transfer_outputs(self, pattern="*.bp"):
        """
        Discover energy_transfer .bp outputs in this simulation's directory,
        grouped into an OutputsCollection keyed by their base filename (the
        part before the .<NNNNN>.bp iteration suffix).

        Unlike `outputs` above, this isn't read off a <parthenon/outputN>
        config section -- the files come from the offline driver (its own
        <energy_transfer> block, output_file/output_number commonly
        overridden at invocation time) or from in-situ WriteResult calls,
        so the input deck alone doesn't reliably say what got written.
        Instead this globs `pattern` in the simulation directory and keeps
        whichever matches energy_transfer::WriteResult wrote -- recognized
        by its donor_shell_edges attribute or a *_pow_sum spectrum record
        (see io_utils.energy_transfer_file_info), which also distinguishes
        them from an ordinary Parthenon openpmd output sharing the same .bp
        extension (e.g. a slice).
        """
        handler = next(
            (h for h in self.output_types if isinstance(h, EnergyTransferOutputType)), None
        )
        if handler is None:
            handler = EnergyTransferOutputType()

        groups = defaultdict(list)
        for path in sorted(glob.glob(os.path.join(self.directory, pattern))):
            fname = os.path.basename(path)
            is_et, time = io_utils.energy_transfer_file_info(path)
            if not is_et:
                continue
            m = re.match(r"^(.*)\.(\d+)\.bp$", fname)
            if m:
                base, idx = m.group(1), int(m.group(2))
            else:
                base, idx = (fname[:-3] if fname.endswith(".bp") else fname), None
            groups[base].append((idx, time, fname))

        result = OutputsCollection()
        for base, files in groups.items():
            files.sort(key=lambda f: (f[0] is None, f[0]))
            result[base] = OutputSeries(
                base, f"energy_transfer [{base}]", files, handler, self.directory
            )
        return result

    def _display_history_widget(self):
        from IPython.display import display, HTML
        import ipywidgets as widgets

        try:
            df = self.history
        except Exception as e:
            display(HTML(f"<em>Could not load history: {e}</em>"))
            return

        cols = [c for c in df.columns if c not in ("time", "redshift")]
        has_redshift = "redshift" in df.columns
        x_options = ["time", "redshift"] if has_redshift else ["time"]

        col_sel = widgets.ToggleButtons(
            options=cols, value=cols[0],
            style={"button_width": "auto"},
        )
        x_sel = widgets.ToggleButtons(
            options=x_options, value="time",
            style={"button_width": "auto"},
        ) if has_redshift else None

        plot_out = widgets.Output()

        def render(col, x):
            fig = Figure(figsize=(6, 2.5))
            ax = fig.add_subplot(111)
            ax.plot(df[x], df[col])
            ax.set_xlabel(x)
            ax.set_ylabel(col)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            return (f"<img src='data:image/png;base64,"
                    f"{base64.b64encode(buf.read()).decode()}'"
                    f" style='max-width:600px'/>")

        def update(change=None):
            x = x_sel.value if x_sel else "time"
            plot_out.clear_output(wait=True)
            with plot_out:
                display(HTML(render(col_sel.value, x)))

        col_sel.observe(update, names="value")
        if x_sel:
            x_sel.observe(update, names="value")

        update()

        hdr = "font-family:monospace;font-weight:bold;margin:10px 0 4px 0;color:#333"
        display(HTML(f"<div style='{hdr}'>History</div>"))
        controls = [col_sel] + ([x_sel] if x_sel else [])
        display(widgets.VBox(controls + [plot_out]))

    def _elapsed_time(self):
        try:
            return self.history["time"].iloc[-1]
        except Exception:
            return None

    def _info_html(self):
        problem_id = self.config.get("job", {}).get("problem_id", "—")
        comment = self.config.get("comment", {}).get("problem", "")
        elapsed = self._elapsed_time()
        rows = [
            ("Problem ID",   problem_id),
            ("Comment",      comment),
            ("Directory",    self.directory),
            ("Elapsed time", f"{elapsed:.4g}" if elapsed is not None else "—"),
        ]
        return (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em'>"
            + "".join(
                f"<tr><td style='padding:3px 12px;color:#555;font-weight:bold'>{k}</td>"
                f"<td style='padding:3px 12px'>{v}</td></tr>"
                for k, v in rows
            )
            + "</table>"
        )

    def _outputs_file_list_html(self):
        hidden = {h.file_type for h in self.output_types if not h.show_in_file_list}
        blocks = []
        for name, label, file_type, files in self._output_info():
            if file_type == "hst" or file_type in hidden:
                continue
            inner_summary = (
                f"{name} &nbsp;<span style='color:#555'>{label}</span>"
                f"&nbsp; ({len(files)} files)"
            )
            file_rows = "".join(
                f"<tr><td>{idx if idx is not None else '—'}</td>"
                f"<td>{f'{t:.4g}' if t is not None else '—'}</td>"
                f"<td>{fname}</td></tr>"
                for idx, t, fname in files
            )
            inner_content = (
                f"<table style='margin:4px 0 4px 16px;border-collapse:collapse;"
                f"font-family:monospace;font-size:0.85em'>"
                f"<tr style='color:#888'><th style='padding:2px 12px'>index</th>"
                f"<th style='padding:2px 12px'>time</th>"
                f"<th style='padding:2px 12px'>file</th></tr>"
                f"{file_rows}</table>"
            )
            blocks.append(
                f"<details style='margin:2px 0'>"
                f"<summary style='font-family:monospace;cursor:pointer'>{inner_summary}</summary>"
                f"<div style='margin-left:8px;margin-top:4px'>{inner_content}</div>"
                f"</details>"
            )
        return "\n".join(blocks)

    def _ipython_display_(self, **kwargs):
        from IPython.display import display, HTML
        import ipywidgets as widgets

        self.close()  # dispose of widgets from a previous render before building new ones

        hdr = "font-family:monospace;font-weight:bold;margin:10px 0 4px 0;color:#333"
        display(HTML(self._info_html()))
        self._display_history_widget()

        all_outputs = self._output_info()
        outputs_by_type = {}
        for name, label, ft, files in all_outputs:
            outputs_by_type.setdefault(ft, []).append((name, label, files))

        accordions = []
        for handler in self.output_types:
            if handler.file_type == "hst":
                continue
            group = outputs_by_type.get(handler.file_type, [])
            if not group:
                continue
            acc = handler.build_accordion(group, self.directory)
            if acc is not None:
                accordions.append(acc)

        config_out = widgets.Output()
        with config_out:
            display(HTML(self.config._repr_html_()))
        config_acc = widgets.Accordion(children=[config_out], selected_index=None)
        config_acc.set_title(0, 'Config')
        accordions.append(config_acc)

        outputs_out = widgets.Output()
        with outputs_out:
            display(HTML(self._outputs_file_list_html()))
        outputs_acc = widgets.Accordion(children=[outputs_out], selected_index=None)
        outputs_acc.set_title(0, 'Outputs')
        accordions.append(outputs_acc)

        vbox = widgets.VBox(accordions)
        self._live_widgets.append(vbox)
        display(vbox)

    def _repr_html_(self):
        try:
            df = self.history
            cols = [c for c in df.columns if c not in ("time", "redshift")]
            fig = Figure(figsize=(6, 2.5))
            ax = fig.add_subplot(111)
            if cols:
                ax.plot(df["time"], df[cols[0]])
                ax.set_xlabel("time")
                ax.set_ylabel(cols[0])
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            hist_html = (
                f"<img src='data:image/png;base64,"
                f"{base64.b64encode(buf.read()).decode()}'"
                f" style='max-width:600px'/>"
            )
        except Exception as e:
            hist_html = f"<em>Could not load history: {e}</em>"
        return self._info_html() + hist_html

    def __repr__(self):
        problem_id = self.config.get("job", {}).get("problem_id", "—")
        comment = self.config.get("comment", {}).get("problem", "")
        elapsed = self._elapsed_time()
        elapsed_str = f"{elapsed:.4g}" if elapsed is not None else "—"
        output_lines = "\n".join(
            f"  {name:<10} {label:<25} {len(files)} files"
            for name, label, file_type, files in self._output_info()
            if file_type != "hst"
        )
        return (
            f"Simulation: {problem_id}\n"
            f"  {comment}\n"
            f"  Directory:    {self.directory}\n"
            f"  Elapsed time: {elapsed_str}\n"
            f"  Outputs:\n{output_lines}"
        )

    @property
    def history(self):
        hist_files = sorted(glob.glob(os.path.join(self.directory, "*.hst")))
        if not hist_files:
            raise FileNotFoundError(f"No history file (*.hst) found in {self.directory}")
        if len(hist_files) > 1:
            print(f"Warning: multiple history files found, using {hist_files[0]}")
        hist_file = hist_files[0]
        try:
            df = pd.read_csv(hist_file, sep=r"\s+", comment="#", header=None)
        except Exception:
            df = pd.read_csv(hist_file, sep=r"\s+", comment="#", engine="python", header=None)
        with open(hist_file) as f:
            for line in f:
                if line.startswith("# ["):
                    columns = [e.split("=")[1].strip() for e in line.split() if "=" in e]
                    break
        df.columns = columns
        df = self._postprocess_history(df)
        return History(df)

    def _domain_volume(self):
        mesh = self.config["parthenon/mesh"]
        vol = 1.0
        for i in (1, 2, 3):
            vol *= float(mesh[f"x{i}max"]) - float(mesh[f"x{i}min"])
        return vol

    def _postprocess_history(self, df):
        """Hook to modify the history DataFrame (e.g. add derived columns).

        No-op in the core; override (e.g. in a physics mixin) to customise.
        Must return the DataFrame.
        """
        return df
