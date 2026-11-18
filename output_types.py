import io
import os
import base64

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

from . import io_utils
from .interface import (
    _EXT_MAP,
    _SPINNER_HTML,
    _lazy_accordion,
    _lazy_widget_accordion,
)

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class History:
    """Wrapper around the history DataFrame with value lookups.

    Behaves like the underlying DataFrame (indexing, ``columns``, ``iloc``,
    ...), plus :meth:`at` to look up one quantity at a given value of another.
    """

    def __init__(self, df):
        self.df = df

    def at(self, quantity, by, value, method="interp"):
        """Value(s) of ``quantity`` where column ``by`` equals ``value``.

        Parameters
        ----------
        quantity : str or list of str
            Column(s) to look up.
        by : str
            Column to search along (e.g. "time", "redshift", or any quantity).
        value : float or array-like
            Target value(s) of ``by``.
        method : "interp" or "nearest"
            "interp": linear interpolation at the first point where ``by``
            crosses ``value`` (works for non-monotonic columns too).
            "nearest": the row whose ``by`` is closest to ``value``.

        Returns a float (scalar value, one quantity), an array (array value),
        or a DataFrame/Series when several quantities are requested.
        Raises ValueError if ``value`` is never reached.
        """
        if method not in ("interp", "nearest"):
            raise ValueError(f"method must be 'interp' or 'nearest', got {method!r}")
        if by not in self.df.columns:
            raise KeyError(f"Unknown column {by!r}; available: {list(self.df.columns)}")
        single_q = isinstance(quantity, str)
        qs = [quantity] if single_q else list(quantity)
        for q in qs:
            if q not in self.df.columns:
                raise KeyError(f"Unknown column {q!r}; available: {list(self.df.columns)}")

        x = self.df[by].to_numpy(dtype=float)
        ys = {q: self.df[q].to_numpy(dtype=float) for q in qs}
        targets = np.atleast_1d(np.asarray(value, dtype=float))

        out = {q: np.empty(len(targets)) for q in qs}
        for k, v in enumerate(targets):
            if method == "nearest":
                i = int(np.nanargmin(np.abs(x - v)))
                for q in qs:
                    out[q][k] = ys[q][i]
                continue
            d = x - v
            hit = np.flatnonzero(d == 0)
            if hit.size:
                i = hit[0]
                for q in qs:
                    out[q][k] = ys[q][i]
                continue
            cross = np.flatnonzero(d[:-1] * d[1:] < 0)
            if not cross.size:
                raise ValueError(
                    f"{by} = {v} is outside the range of the data "
                    f"[{np.nanmin(x)}, {np.nanmax(x)}]"
                )
            i = cross[0]
            f = (v - x[i]) / (x[i + 1] - x[i])
            for q in qs:
                out[q][k] = ys[q][i] + f * (ys[q][i + 1] - ys[q][i])

        scalar = np.ndim(value) == 0
        if single_q:
            return float(out[quantity][0]) if scalar else out[quantity]
        if scalar:
            return pd.Series({q: out[q][0] for q in qs})
        return pd.DataFrame(out, index=targets)

    # --- DataFrame-like behaviour -----------------------------------------
    def __getitem__(self, key):
        return self.df[key]

    def __getattr__(self, name):
        if name == "df":  # avoid recursion before __init__ / during unpickling
            raise AttributeError(name)
        return getattr(self.df, name)

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        return iter(self.df)

    def __repr__(self):
        return repr(self.df)

    def _repr_html_(self):
        return self.df._repr_html_()


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


class HistogramOutputType(OutputType):
    """Parthenon histogram outputs (``parthenon.out<n>.histograms.<NNNNN>.hdf``).

    ``frame.data`` is a dict mapping histogram name -> dict with keys
    ``data``, ``x_edges`` (and ``y_edges`` for 2D), ``x_centers``, plus the
    HDF5 attributes (``x_var_name``, ``binned_var_name``, ...). ``frame[name]``
    returns that single histogram's dict.
    """
    file_type = "histogram"
    accordion_title = "Histograms"

    def file_pattern(self, n, cfg):
        return (f"parthenon.out{n}.histograms.*{_EXT_MAP['histogram']}",
                f"histogram [out{n}]")

    def read_data(self, path):
        import h5py
        result = {}
        with h5py.File(path, "r") as f:
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f["Info"].attrs.get("hist_names", list(f.keys()))]
            for name in names:
                if name not in f:
                    continue
                g = f[name]
                h = {k: (v.decode() if isinstance(v, bytes) else
                         v.item() if hasattr(v, "item") else v)
                     for k, v in g.attrs.items()}
                h["data"] = g["data"][()]
                for ax in ("x", "y"):
                    key = f"{ax}_edges"
                    if key in g:
                        h[key] = g[key][()]
                        h[f"{ax}_centers"] = 0.5 * (h[key][1:] + h[key][:-1])
                result[name] = h
        return result

    def _plot(self, h, name, ax):
        if h.get("ndim", 1) == 2 and "y_edges" in h:
            im = ax.pcolormesh(h["x_edges"], h["y_edges"], h["data"].T, cmap="viridis")
            ax.figure.colorbar(im, ax=ax)
            ax.set_ylabel(h.get("y_var_name", "y"))
        else:
            ax.stairs(h["data"], h["x_edges"])
            ax.set_ylabel(h.get("binned_var_name", "count"))
        ax.set_xlabel(h.get("x_var_name", "x"))
        ax.set_title(name)

    def frame_repr_html(self, frame):
        hists = frame.data
        n = max(len(hists), 1)
        fig = Figure(figsize=(4.5 * n, 3.2))
        for i, (name, h) in enumerate(hists.items()):
            self._plot(h, name, fig.add_subplot(1, n, i + 1))
        t_str = f"{frame.time:.4g}" if frame.time is not None else "—"
        fig.suptitle(f"t = {t_str}")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        return (f"<img src='data:image/png;base64,"
                f"{base64.b64encode(buf.read()).decode()}'/>")

    def get_frame_field(self, frame, key):
        return frame.data[key]

    def build_accordion(self, outputs, directory):
        return None


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
