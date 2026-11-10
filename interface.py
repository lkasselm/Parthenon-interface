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

import numpy as np
import pandas as pd
import glob
import os
from . import io_utils
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

_EXT_MAP = {
    "hst":      ".hst",
    "hdf5":     ".phdf",
    "rst":      ".rhdf",
    "openpmd":  ".bp",
    "spectrum": ".spc",
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


# ---------------------------------------------------------------------------
# OutputFrame and OutputSeries
# ---------------------------------------------------------------------------

class YTFieldFrame:
    """One field from one Parthenon HDF5 timestep, backed by a yt dataset."""

    def __init__(self, field, frame, axis="z"):
        self.field = field
        self.time = frame.time
        self._frame = frame
        self._axis = axis
        self._data_cache = None

    @property
    def data(self):
        """3-D numpy array for this field (covering grid at level 0)."""
        if self._data_cache is None:
            ds = self._frame.data
            cg = ds.covering_grid(
                level=0,
                left_edge=ds.domain_left_edge,
                dims=ds.domain_dimensions,
            )
            self._data_cache = np.array(cg[self.field])
        return self._data_cache

    def _repr_html_(self):
        import yt
        yt.set_log_level("error")
        slc = yt.SlicePlot(self._frame.data, self._axis, self.field)
        slc.set_log(self.field, False)
        slc.set_cmap(self.field, "viridis")
        return slc._repr_html_()

    def __repr__(self):
        t_str = f"{self.time:.4g}" if self.time is not None else "—"
        return f"YTFieldFrame(field={self.field!r}, t={t_str})"


class FieldFrame:
    """One field from one OpenPMD timestep."""

    def __init__(self, field, array, extent, time):
        self.field = field
        self.time = time
        self.extent = extent
        self._array = array

    @property
    def data(self):
        return self._array

    def _repr_html_(self):
        fig = Figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111)
        im = ax.imshow(self._array, origin="lower", extent=self.extent, cmap="viridis")
        fig.colorbar(im, ax=ax, label=self.field)
        t_str = f"{self.time:.4g}" if self.time is not None else "—"
        ax.set_title(f"{self.field},  t = {t_str}")
        ax.set_xlabel("x2")
        ax.set_ylabel("x3")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        return (f"<img src='data:image/png;base64,"
                f"{base64.b64encode(buf.read()).decode()}'"
                f" style='max-width:600px'/>")

    def __repr__(self):
        t_str = f"{self.time:.4g}" if self.time is not None else "—"
        return f"FieldFrame(field={self.field!r}, t={t_str}, shape={self._array.shape})"


class EnergyTransferResult:
    """
    One energy_transfer output, as written by the C++ WriteResult: shell-
    transfer matrices (`.terms`, plus the donor/mediator/receiver binning
    that produced them), spectra (`.spectra`), or both -- WriteResult's two
    parts are each optional (see io_utils.parse_energy_transfer_file for the
    on-disk format this wraps).

    Read directly from a file, no Simulation needed:

        result = EnergyTransferResult.from_file("mediator_q23_output.00002.bp")

    Or discovered from a Simulation (see Simulation.energy_transfer_outputs):

        result = sim.energy_transfer_outputs()["mediator_q23_output"][0].data

    Each entry in `.terms` is an array shaped (n_receiver, n_mediator,
    n_donor) -- receiver (K) outermost/rows, donor (Q) innermost/cols,
    mediator in between, matching this package's existing (K, Q)
    shell-matrix convention (see io_utils.dict_to_array) with the mediator
    axis sandwiched in. An axis energy_transfer didn't resolve (collapsed to
    the whole domain) has length 1 -- index/squeeze it out if not needed,
    e.g. `result["UUA"][:, 0, :]` for the plain (K, Q) matrix when the
    mediator wasn't decomposed.

    Each entry in `.spectra` is the raw {"pow_sum", "k_sum", "count_sum"}
    triple per unit-width |k| bin; `result.spectrum("spec_U")` returns it as
    a DataFrame with the bin-mean wavenumber k = k_sum / count_sum added.
    `result[name]` looks up a term first, then a spectrum.
    """

    def __init__(self, terms, spectra, donor_edges, mediator_edges, receiver_edges,
                 donor_binning, mediator_binning, receiver_binning, time=None):
        self.terms = terms
        self.spectra = spectra
        # Edges/binning are None for a spectra-only output.
        self.donor_edges = None if donor_edges is None else np.asarray(donor_edges)
        self.mediator_edges = None if mediator_edges is None else np.asarray(mediator_edges)
        self.receiver_edges = None if receiver_edges is None else np.asarray(receiver_edges)
        self.donor_binning = donor_binning
        self.mediator_binning = mediator_binning
        self.receiver_binning = receiver_binning
        self.time = time

    @classmethod
    def from_file(cls, path):
        """Read an energy_transfer .bp output directly -- no Simulation needed."""
        return cls(**io_utils.parse_energy_transfer_file(path))

    def __getitem__(self, name):
        if name in self.terms:
            return self.terms[name]
        if name in self.spectra:
            return self.spectrum(name)
        raise KeyError(f"{name!r} is neither a term nor a spectrum in this output "
                       f"(terms: {list(self.terms)}, spectra: {list(self.spectra)})")

    def __contains__(self, name):
        return name in self.terms or name in self.spectra

    def __iter__(self):
        yield from self.terms
        yield from self.spectra

    def spectrum(self, name):
        """
        Spectrum `name` as a DataFrame with columns pow_sum, k_sum, count_sum
        and k (= k_sum / count_sum, the bin-mean wavenumber; NaN for an empty
        bin). Index is the bin number.
        """
        raw = self.spectra[name]
        df = pd.DataFrame({key: np.asarray(raw[key]) for key in ("pow_sum", "k_sum", "count_sum")})
        with np.errstate(invalid="ignore", divide="ignore"):
            df["k"] = np.where(df["count_sum"] > 0, df["k_sum"] / df["count_sum"], np.nan)
        return df

    @staticmethod
    def _n_shells(edges):
        return None if edges is None else len(edges) - 1

    @property
    def n_donor(self):
        return self._n_shells(self.donor_edges)

    @property
    def n_mediator(self):
        return self._n_shells(self.mediator_edges)

    @property
    def n_receiver(self):
        return self._n_shells(self.receiver_edges)

    @staticmethod
    def _edge_labels(edges):
        return [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(edges[:-1], edges[1:])]

    @staticmethod
    def _fig_to_img(fig, max_width):
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        return (f"<img src='data:image/png;base64,"
                f"{base64.b64encode(buf.read()).decode()}'"
                f" style='max-width:{max_width}px'/>")

    def _terms_html(self):
        names = list(self.terms)
        ncols = min(4, len(names))
        nrows = -(-len(names) // ncols)
        fig = Figure(figsize=(3.2 * ncols, 2.8 * nrows))
        axes = fig.subplots(nrows, ncols, squeeze=False)
        donor_labels = self._edge_labels(self.donor_edges)
        receiver_labels = self._edge_labels(self.receiver_edges)
        for i, name in enumerate(names):
            ax = axes[i // ncols][i % ncols]
            # Sum over the mediator axis for this at-a-glance grid -- index
            # .terms[name] directly to look at individual mediator shells.
            summed = self.terms[name].sum(axis=1)
            scale = np.max(np.abs(summed)) or 1.0
            im = ax.imshow(summed, cmap="RdBu_r", vmin=-scale, vmax=scale, origin="lower")
            ax.set_title(name, fontsize=9)
            if len(donor_labels) <= 12:
                ax.set_xticks(range(len(donor_labels)))
                ax.set_xticklabels(donor_labels, rotation=90, fontsize=5)
                ax.set_yticks(range(len(receiver_labels)))
                ax.set_yticklabels(receiver_labels, fontsize=5)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
        for j in range(len(names), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        t_str = f", t = {self.time:.4g}" if self.time is not None else ""
        fig.suptitle(f"energy_transfer{t_str} (summed over {self.n_mediator} mediator shell(s))")
        return self._fig_to_img(fig, 900)

    def _spectra_html(self):
        fig = Figure(figsize=(6, 3.5))
        ax = fig.add_subplot(111)
        y_max = None
        for name in self.spectra:
            df = self.spectrum(name)
            ax.loglog(df["k"], df["pow_sum"], linewidth=0.8, label=name)
            positive = df["pow_sum"][df["pow_sum"] > 0]
            if not positive.empty and (y_max is None or positive.max() > y_max):
                y_max = positive.max()
        if y_max is not None and np.isfinite(y_max):
            log_max = np.log10(y_max)
            ax.set_ylim(10 ** (log_max - 8), 10 ** (log_max + 1))
        ax.set_xlabel("k (bin mean)")
        ax.set_ylabel("pow_sum")
        ax.legend(fontsize=6)
        t_str = f"t = {self.time:.4g}" if self.time is not None else ""
        ax.set_title(f"energy_transfer spectra {t_str}".strip(), fontsize=9)
        return self._fig_to_img(fig, 600)

    def _repr_html_(self):
        parts = []
        if self.terms:
            parts.append(self._terms_html())
        if self.spectra:
            parts.append(self._spectra_html())
        if not parts:
            return "<em>No terms or spectra in this energy_transfer output.</em>"
        return "<br/>".join(parts)

    def __repr__(self):
        t_str = f"t={self.time:.4g}" if self.time is not None else "t=—"
        shape_str = ""
        if self.terms:
            shape_str = (f", shape=(n_receiver={self.n_receiver}, n_mediator={self.n_mediator}, "
                         f"n_donor={self.n_donor})")
        return (f"EnergyTransferResult({t_str}, terms={list(self.terms)}, "
                f"spectra={list(self.spectra)}{shape_str})")


class OutputFrame:
    """A single timestep of one output series."""

    def __init__(self, idx, time, path, handler):
        self.idx = idx
        self.time = time
        self.path = path
        self._handler = handler
        self._data_cache = None

    @property
    def data(self):
        """Raw data from this frame (cached). Type depends on the output handler."""
        if self._data_cache is None:
            self._data_cache = self._handler.read_data(self.path)
        return self._data_cache

    def __getitem__(self, key):
        return self._handler.get_frame_field(self, key)

    def _repr_html_(self):
        return self._handler.frame_repr_html(self)

    def __repr__(self):
        t_str = f"{self.time:.4g}" if self.time is not None else "—"
        return f"OutputFrame(idx={self.idx}, t={t_str}, file={os.path.basename(self.path)})"


class OutputSeries:
    """All frames for one named output block (e.g. all sl.x1.0.5 files)."""

    def __init__(self, name, label, files, handler, directory):
        self.name = name
        self.label = label
        self._files = files          # [(idx, time, fname), ...]
        self._handler = handler
        self._directory = directory

    def __len__(self):
        return len(self._files)

    def __getitem__(self, i):
        idx, time, fname = self._files[i]
        return OutputFrame(idx, time, os.path.join(self._directory, fname), self._handler)

    def at(self, t):
        """Return the frame whose time is closest to t."""
        best = min(
            range(len(self._files)),
            key=lambda i: abs(self._files[i][1] - t) if self._files[i][1] is not None else float('inf'),
        )
        return self[best]

    def times(self):
        """Return a list of frame times."""
        return [t for _, t, _ in self._files]

    def __repr__(self):
        return f"OutputSeries({self.label}, {len(self)} frames)"


class OutputsCollection(dict):
    """Dict of OutputSeries with a rich Jupyter HTML repr."""

    def _repr_html_(self):
        rows = []
        for key, series in self.items():
            ft = series._handler.file_type if series._handler else "?"
            times = [t for t in series.times() if t is not None]
            t_range = (f"{min(times):.3g} – {max(times):.3g}" if times else "—")
            rows.append(
                f"<tr>"
                f"<td style='padding:4px 14px;font-weight:bold'>{key}</td>"
                f"<td style='padding:4px 14px;color:#555'>{ft}</td>"
                f"<td style='padding:4px 14px'>{len(series)}</td>"
                f"<td style='padding:4px 14px'>{t_range}</td>"
                f"</tr>"
            )
        header = (
            "<tr style='border-bottom:1px solid #ccc;color:#888'>"
            "<th style='padding:4px 14px;font-weight:normal;text-align:left'>id</th>"
            "<th style='padding:4px 14px;font-weight:normal;text-align:left'>type</th>"
            "<th style='padding:4px 14px;font-weight:normal;text-align:left'>frames</th>"
            "<th style='padding:4px 14px;font-weight:normal;text-align:left'>time range</th>"
            "</tr>"
        )
        return (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em'>"
            + header + "".join(rows) + "</table>"
        )


# ---------------------------------------------------------------------------
# OutputType base class and built-in handlers
# ---------------------------------------------------------------------------

class OutputType:
    """
    Base class for a Parthenon output type.

    Subclass and set file_type + accordion_title, then implement
    file_pattern() and build_accordion(). Register via
    sim.output_types.append(MyOutputType()).
    """
    file_type: str = ""
    accordion_title: str = ""
    show_in_file_list: bool = True

    def file_pattern(self, n: str, cfg: dict) -> tuple:
        """Return (glob_pattern, label) for locating files of this type."""
        raise NotImplementedError

    def build_accordion(self, outputs: list, directory: str):
        """
        outputs: [(name, label, [(idx, time, fname), ...]), ...]
        Return a lazy widgets.Accordion, or None to skip.
        """
        raise NotImplementedError

    def read_data(self, path):
        """Read raw data from a single file. Return type is handler-specific."""
        raise NotImplementedError

    def frame_repr_html(self, frame) -> str:
        """Return an HTML string (embedded PNG) for a single OutputFrame."""
        raise NotImplementedError

    def get_frame_field(self, frame, key):
        """Return a field-specific object for frame[key]. Override in handlers that support it."""
        raise TypeError(f"{type(self).__name__} does not support field indexing")


class SpectrumOutputType(OutputType):
    file_type = "spectrum"
    accordion_title = "Spectra"

    def __init__(self):
        self._cache = {}

    def file_pattern(self, n, cfg):
        output_label = cfg.get("output_label")
        if output_label is None:
            # Older builds without output_label write parthenon.out<n>.<NNNNN>.spc
            return (f"parthenon.out{n}.*{_EXT_MAP['spectrum']}",
                    f"spectrum [out{n}]")
        return (f"parthenon.{output_label}.out{n}.*{_EXT_MAP['spectrum']}",
                f"spectrum [{output_label}]")

    def _render_one(self, name, files, directory):
        if name in self._cache:
            return self._cache[name]
        times = [t for _, t, _ in files]
        t_vals = [t for t in times if t is not None]
        t_min, t_max = (min(t_vals), max(t_vals)) if t_vals else (0, 1)
        norm = mcolors.Normalize(vmin=t_min, vmax=t_max)
        cmap = matplotlib.colormaps["viridis"]
        fig = Figure(figsize=(6, 3.5))
        ax = fig.add_subplot(111)
        y_max = None
        for i, (idx, t, fname) in enumerate(files):
            fpath = os.path.join(directory, fname)
            try:
                df = io_utils.parse_spc_file(fpath)
                color = cmap(norm(t)) if t is not None else cmap(i / max(len(files) - 1, 1))
                ax.loglog(df["Bin"], df["En_sum"], color=color, linewidth=0.8)
                positive = df["En_sum"][df["En_sum"] > 0]
                if not positive.empty:
                    col_max = positive.max()
                    if np.isfinite(col_max) and (y_max is None or col_max > y_max):
                        y_max = col_max
            except Exception:
                pass
        if y_max is not None and np.isfinite(y_max) and y_max > 0:
            log_max = np.log10(y_max)
            ax.set_ylim(10 ** (log_max - 5), 10 ** (log_max + 1))
        ax.set_xlabel("bin")
        ax.set_ylabel("En_sum")
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="time")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        result = (f"<img src='data:image/png;base64,"
                  f"{base64.b64encode(buf.read()).decode()}'"
                  f" style='max-width:600px'/>")
        self._cache[name] = result
        return result

    def _render_section(self, outputs, directory):
        suid = f"spectra_{abs(hash(directory))}"
        btn_style = ("font-family:monospace;padding:3px 8px;margin:2px;"
                     "cursor:pointer;border:1px solid #ccc;border-radius:3px")
        buttons, plot_divs = [], []
        for i, (name, out_label, files) in enumerate(outputs):
            short = out_label.replace("spectrum [", "").rstrip("]")
            onclick = (
                f"document.querySelectorAll('.{suid}-plot')"
                f".forEach(function(e){{e.style.display='none'}});"
                f"document.querySelectorAll('.{suid}-btn')"
                f".forEach(function(e){{e.style.background='#f0f0f0';e.style.color='black'}});"
                f"document.getElementById('{suid}-{i}').style.display='block';"
                f"this.style.background='#333';this.style.color='white';"
            )
            bg = "#333;color:white" if i == 0 else "#f0f0f0"
            buttons.append(
                f"<button class='{suid}-btn' onclick=\"{onclick}\""
                f" style='{btn_style};background:{bg}'>{short}</button>"
            )
            plot_divs.append(
                f"<div id='{suid}-{i}' class='{suid}-plot'"
                f" style='display:{'block' if i == 0 else 'none'}'>"
                f"{self._render_one(name, files, directory)}</div>"
            )
        return (
            f"<div style='margin-bottom:6px'>{''.join(buttons)}</div>"
            + "".join(plot_divs)
        )

    def read_data(self, path):
        return io_utils.parse_spc_file(path)

    def frame_repr_html(self, frame):
        df = frame.data
        fig = Figure(figsize=(6, 3.5))
        ax = fig.add_subplot(111)
        positive = df["En_sum"][df["En_sum"] > 0]
        ax.loglog(df["Bin"], df["En_sum"], linewidth=0.8)
        if not positive.empty:
            log_max = np.log10(positive.max())
            ax.set_ylim(10 ** (log_max - 5), 10 ** (log_max + 1))
        t_str = f"{frame.time:.4g}" if frame.time is not None else "—"
        ax.set_title(f"t = {t_str}")
        ax.set_xlabel("bin")
        ax.set_ylabel("En_sum")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        return (f"<img src='data:image/png;base64,"
                f"{base64.b64encode(buf.read()).decode()}'"
                f" style='max-width:600px'/>")

    def build_accordion(self, outputs, directory):
        return _lazy_accordion(
            self.accordion_title,
            lambda: self._render_section(outputs, directory),
        )


def _opmd_field_map(it, opmd):
    """Map display label -> (mesh_name, component) for every field in an iteration.

    Parthenon writes one openPMD mesh per (packed-variable, component) pair, so a
    scalar field like "density" ends up as its own mesh whose only record component
    is the generic openPMD SCALAR sentinel (not the field name). Older writers instead
    grouped everything under one mesh with the real field names as components. Handle
    both: use the mesh's own name as the label when its only component is the SCALAR
    sentinel, otherwise use the component name (which is already the real field name
    unless it's a generic vector component, in which case prefix it with the mesh name).
    """
    scalar = opmd.Mesh_Record_Component.SCALAR
    field_map = {}
    for mesh_name in it.meshes:
        mesh = it.meshes[mesh_name]
        comps = list(mesh)
        if comps == [scalar]:
            field_map[mesh_name] = (mesh_name, scalar)
        else:
            for comp in comps:
                comp_str = str(comp)
                label = f"{mesh_name}_{comp_str}" if comp_str in ("x", "y", "z") else comp_str
                field_map[label] = (mesh_name, comp)
    return field_map


class OpenPMDOutputType(OutputType):
    file_type = "openpmd"
    accordion_title = "Slices"

    def file_pattern(self, n, cfg):
        oid = cfg.get("id", f"out{n}")
        return f"parthenon.{oid}.*{_EXT_MAP['openpmd']}", f"openpmd [{oid}]"

    def _display_widget(self, outputs, directory):
        from IPython.display import display, HTML
        import ipywidgets as widgets
        try:
            import openpmd_api as opmd
        except ImportError:
            display(HTML("<em>openpmd-api not installed. Run: pip install openpmd-api</em>"))
            return

        MAX_FRAMES = 15

        def make_panel(files):
            if len(files) > MAX_FRAMES:
                idxs = np.linspace(0, len(files) - 1, MAX_FRAMES, dtype=int)
                sampled = [files[i] for i in idxs]
            else:
                sampled = list(files)
            time_labels = [f"{t:.3g}" if t is not None else str(idx)
                           for idx, t, _ in sampled]

            first_path = os.path.join(directory, sampled[0][2])
            try:
                s = opmd.Series(first_path, opmd.Access.read_only)
                it_num = next(iter(s.iterations))
                it = s.iterations[it_num]
                field_map = _opmd_field_map(it, opmd)
                fields = list(field_map)
                mesh0_name, comp0 = field_map[fields[0]]
                mesh0 = it.meshes[mesh0_name]
                dx = list(mesh0.grid_spacing)
                x0 = list(mesh0.grid_global_offset)
                shape = mesh0[comp0].shape
                extent = [x0[1], x0[1] + dx[1]*shape[1],
                          x0[0], x0[0] + dx[0]*shape[0]]
                s.close()
            except Exception as e:
                display(HTML(f"<em>Error reading metadata: {e}</em>"))
                return None

            field_sel = widgets.ToggleButtons(
                options=fields, value=fields[0],
                style={"button_width": "auto"})
            slider = widgets.IntSlider(
                min=0, max=len(sampled) - 1, value=0,
                layout=widgets.Layout(width="400px"))
            time_label = widgets.Label(value=f"t = {time_labels[0]}")
            plot_out = widgets.Output()

            def render(field, ti):
                _, _, fname = sampled[ti]
                fpath = os.path.join(directory, fname)
                try:
                    s = opmd.Series(fpath, opmd.Access.read_only)
                    it_num = next(iter(s.iterations))
                    it = s.iterations[it_num]
                    mesh_name, comp = _opmd_field_map(it, opmd)[field]
                    mesh = it.meshes[mesh_name]
                    chunk = mesh[comp].load_chunk()
                    s.flush()
                    data = chunk.copy()
                    s.close()
                    fig = Figure(figsize=(6, 4.5))
                    ax = fig.add_subplot(111)
                    im = ax.imshow(data, origin="lower", extent=extent, cmap="viridis")
                    fig.colorbar(im, ax=ax, label=field)
                    ax.set_title(f"t = {time_labels[ti]}")
                    ax.set_xlabel("x2")
                    ax.set_ylabel("x3")
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100)
                    buf.seek(0)
                    return (f"<img src='data:image/png;base64,"
                            f"{base64.b64encode(buf.read()).decode()}'"
                            f" style='max-width:600px'/>")
                except Exception as e:
                    return f"<em>Error: {e}</em>"

            def update(change=None):
                time_label.value = f"t = {time_labels[slider.value]}"
                plot_out.clear_output(wait=True)
                with plot_out:
                    display(HTML(_SPINNER_HTML))
                html = render(field_sel.value, slider.value)
                plot_out.clear_output(wait=True)
                with plot_out:
                    display(HTML(html))

            field_sel.observe(update, names="value")
            slider.observe(update, names="value")
            update()

            return widgets.VBox([
                field_sel,
                widgets.HBox([slider, time_label]),
                plot_out,
            ])

        if len(outputs) == 1:
            _, _, files = outputs[0]
            w = make_panel(files)
            if w:
                display(w)
        else:
            children, titles = [], []
            for name, label, files in outputs:
                w = make_panel(files)
                if w:
                    children.append(w)
                    titles.append(label.replace("openpmd [", "").rstrip("]"))
            if children:
                tab = widgets.Tab(children=children)
                for i, t in enumerate(titles):
                    tab.set_title(i, t)
                display(tab)

    def read_data(self, path):
        try:
            import openpmd_api as opmd
        except ImportError:
            raise ImportError("openpmd-api not installed. Run: pip install openpmd-api")
        s = opmd.Series(path, opmd.Access.read_only)
        it_num = next(iter(s.iterations))
        it = s.iterations[it_num]
        field_map = _opmd_field_map(it, opmd)
        fields = list(field_map)
        mesh0_name, comp0 = field_map[fields[0]]
        mesh0 = it.meshes[mesh0_name]
        dx = list(mesh0.grid_spacing)
        x0 = list(mesh0.grid_global_offset)
        shape = mesh0[comp0].shape
        extent = [x0[1], x0[1] + dx[1]*shape[1], x0[0], x0[0] + dx[0]*shape[0]]
        chunks = {f: it.meshes[mname][comp].load_chunk() for f, (mname, comp) in field_map.items()}
        s.flush()
        data = {f: chunks[f].copy() for f in fields}
        s.close()
        return {"fields": data, "extent": extent}

    def _field_names(self, path):
        try:
            import openpmd_api as opmd
        except ImportError:
            return []
        s = opmd.Series(path, opmd.Access.read_only)
        it_num = next(iter(s.iterations))
        it = s.iterations[it_num]
        fields = list(_opmd_field_map(it, opmd))
        s.close()
        return fields

    def frame_repr_html(self, frame):
        fields = self._field_names(frame.path)
        t_str = f"{frame.time:.4g}" if frame.time is not None else "—"
        field_chips = "".join(
            f"<span style='font-family:monospace;background:#f0f0f0;"
            f"border-radius:3px;padding:2px 8px;margin:2px;display:inline-block'>"
            f"{f}</span>"
            for f in fields
        )
        return (
            f"<div style='font-family:monospace;font-size:0.9em'>"
            f"<b>OutputFrame</b>&nbsp; idx={frame.idx}&nbsp; t={t_str}<br>"
            f"<span style='color:#555'>Fields:</span>&nbsp;{field_chips}"
            f"</div>"
        )

    def get_frame_field(self, frame, key):
        result = frame.data
        if key not in result["fields"]:
            raise KeyError(f"Field {key!r} not found. Available: {list(result['fields'].keys())}")
        return FieldFrame(key, result["fields"][key], result["extent"], frame.time)

    def build_accordion(self, outputs, directory):
        return _lazy_widget_accordion(
            self.accordion_title,
            lambda: self._display_widget(outputs, directory),
        )


class HDF5OutputType(OutputType):
    file_type = "hdf5"
    accordion_title = ""

    def file_pattern(self, n, cfg):
        oid = cfg.get("id", f"out{n}")
        return f"parthenon.{oid}.*{_EXT_MAP['hdf5']}", f"hdf5 [{oid}]"

    def read_data(self, path):
        try:
            import yt
        except ImportError:
            raise ImportError("yt not installed. Run: pip install yt")
        yt.set_log_level("error")
        return yt.load(path)

    def frame_repr_html(self, frame):
        ds = frame.data  # yt dataset (cached on frame after first call)
        t_str = f"{frame.time:.4g}" if frame.time is not None else "—"
        field_chips = "".join(
            f"<span style='font-family:monospace;background:#f0f0f0;"
            f"border-radius:3px;padding:2px 8px;margin:2px;display:inline-block'>"
            f"{f}</span>"
            for f in ds.field_list
        )
        return (
            f"<div style='font-family:monospace;font-size:0.9em'>"
            f"<b>OutputFrame</b> (hdf5) &nbsp; idx={frame.idx} &nbsp; t={t_str}<br>"
            f"<span style='color:#555'>Fields:</span>&nbsp;{field_chips}"
            f"</div>"
        )

    def get_frame_field(self, frame, key):
        return YTFieldFrame(key, frame)

    def build_accordion(self, outputs, directory):
        return None


class EnergyTransferOutputType(OutputType):
    """
    Handler for energy_transfer's own .bp outputs (see EnergyTransferResult
    / io_utils.parse_energy_transfer_file) -- shell transfers, spectra, or
    both per file.

    Unlike the other handlers, this isn't discovered via the usual
    <parthenon/outputN> config scan -- the files come either from the
    offline driver (configured through its own <energy_transfer> block,
    frequently with output_file/output_number overridden at invocation
    time) or from an application calling WriteResult in-situ (e.g. from
    UserWorkBeforeOutput), neither of which an input deck reliably
    describes. Use Simulation.energy_transfer_outputs() to
    discover these from a Simulation (it scans the directory directly
    rather than reading a config section), or EnergyTransferResult.from_file(path)
    to read one standalone, no Simulation involved at all.
    """
    file_type = "energy_transfer"
    accordion_title = "Energy transfer"
    show_in_file_list = False  # not part of the <parthenon/outputN> file list at all

    def file_pattern(self, n, cfg):
        raise NotImplementedError(
            "energy_transfer isn't a <parthenon/outputN> block -- use "
            "Simulation.energy_transfer_outputs() instead."
        )

    def read_data(self, path):
        return EnergyTransferResult.from_file(path)

    def frame_repr_html(self, frame):
        return frame.data._repr_html_()

    def get_frame_field(self, frame, key):
        return frame.data[key]

    def build_accordion(self, outputs, directory):
        return None


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

class Simulation:
    """
    Interface for analyzing Parthenon simulation data.

    Output types are handled by OutputType subclasses stored in
    self.output_types. Append a new handler to support additional file types:

        sim.output_types.append(MyOutputType())
    """

    _default_output_type_classes = [
        SpectrumOutputType,
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
        return df
