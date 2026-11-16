# Simulation Interface

A Jupyter notebook interface for exploring [AthenaPK](https://github.com/parthenon-hpc-lab/athenapk) / [Parthenon](https://github.com/parthenon-hpc-lab/parthenon) simulation output.

## Setup

Install once (editable, so local edits are picked up without reinstalling):

```bash
pip install -e /path/to/interface --no-deps
```

`--no-deps` avoids pip clobbering conda-installed packages like `yt` and
`openpmd-api`; see [Dependencies](#dependencies) below.

Then, from any directory:

```python
from interface import Simulation
```

---

## Creating a Simulation

```python
sim = Simulation("path/to/output/directory")
```

The directory must contain exactly one `.in` input file. All output files are discovered automatically from the input file's `<parthenon/outputN>` sections.

Displaying `sim` in a notebook renders an interactive overview:

- **Info** — problem ID, comment, directory, elapsed time
- **History** — clickable tab strip of all history quantities plotted against time
- **Spectra** — overlaid log-log plots (lazy, opens on click)
- **Slices** — field selector + time slider for OpenPMD slice outputs (lazy, one frame rendered at a time)
- **Config** — collapsible view of the full input file
- **Outputs** — file list with index and time for every output

---

## History

```python
h = sim.history    # History: DataFrame-like wrapper, columns = all history quantities + "time"
h["ME"]            # magnetic energy time series (pandas Series)
h.df               # the underlying pandas DataFrame
```

If `KE` and/or `ME` are present, `v_rms` and `B_rms` columns are added
automatically (`E = 1/2 X_rms² V`, with `V` the domain volume from
`parthenon/mesh`). If that fails for any reason the columns are just omitted.

Look up one quantity at a given value of another with `at(quantity, by, value)`:

```python
h.at("KE", "time", 5.0)                       # linear interpolation
h.at("KE", "time", 5.0, method="nearest")     # closest row instead
h.at("time", "KE", 1e3)                       # reverse lookup: when does KE reach 1e3?
h.at(["KE", "ME"], "redshift", 2.0)           # several quantities -> Series
h.at("KE", "time", [1, 2, 3])                 # array of values -> array
```

`by` can be any column. Interpolation uses the first point where `by` crosses
`value`, and raises `ValueError` if the value is never reached.

---

## Outputs

`sim.outputs` returns an `OutputsCollection` — a dict-like object that renders as a summary table in Jupyter showing the output id, type, frame count, and time range.

```python
sim.outputs                  # rendered table in Jupyter
sim.outputs.keys()           # e.g. ["sl.x1.0.5", "B", "v", "prim"]
```

Each value is an `OutputSeries`:

```python
series = sim.outputs["sl.x1.0.5"]
len(series)          # number of frames
series.times()       # list of floats, one per frame
series[0]            # first frame (OutputFrame)
series.at(t=5.0)     # frame closest to t=5.0
```

---

## Output types

### Spectrum (`.spc`)

```python
series = sim.outputs["B"]   # spectrum output with output_label = B
frame  = series.at(t=2.0)

frame              # shows available fields chip list
frame.data         # pandas DataFrame with columns: Bin, En_sum, K_sum, Count
frame              # displays a log-log plot (En_sum vs Bin) in Jupyter
```

### OpenPMD slices (`.bp`)

```python
series = sim.outputs["sl.x1.0.5"]
frame  = series[0]

frame                      # shows available field chips (no data loaded)
frame["density"]           # FieldFrame — renders an imshow
frame["density"].data      # 2-D numpy array
```

`FieldFrame` attributes:
| attribute | description |
|-----------|-------------|
| `.data`   | 2-D numpy array |
| `.field`  | field name string |
| `.time`   | simulation time |
| `.extent` | `[x_min, x_max, y_min, y_max]` in code units |

### HDF5 snapshots / prim (`.phdf`)

```python
series = sim.outputs["prim"]
frame  = series.at(t=5.0)

frame                              # shows available yt fields as chips
frame.data                         # full yt dataset (yt.load result)
frame[("gas", "density")]          # YTFieldFrame — renders a yt SlicePlot
frame[("gas", "density")].data     # 3-D numpy array (covering grid, level 0)
```

`YTFieldFrame` attributes:
| attribute | description |
|-----------|-------------|
| `.data`   | 3-D numpy array |
| `.field`  | yt field tuple |
| `.time`   | simulation time |

The slice is through the `"z"` axis by default (linear scale, viridis colormap).
Use `frame.data` to access the full yt dataset for custom analysis:

```python
ds = frame.data
ds.print_stats()
p = yt.ProjectionPlot(ds, "x", ("gas", "density"))
```

---

## Config

```python
sim.config                          # SimulationConfig (dict subclass)
sim.config["parthenon/mesh"]        # dict of keys in that section
sim.config["parthenon/mesh"]["nx1"] # individual value
```

Displaying `sim.config` in Jupyter renders collapsible sections.

---

## Adding a custom output type

Subclass `OutputType` and register it on the simulation instance:

```python
from interface import OutputType, Simulation

class MyOutputType(OutputType):
    file_type = "my_type"          # matches file_type= in the input file
    accordion_title = "My Output"  # label for the lazy accordion
    show_in_file_list = True       # include in the Outputs file list

    def file_pattern(self, n, cfg):
        oid = cfg.get("id", f"out{n}")
        return f"parthenon.{oid}.*.myext", f"my_type [{oid}]"

    def read_data(self, path):
        # return whatever raw data makes sense
        return np.load(path)

    def frame_repr_html(self, frame):
        # return an HTML string to display a single frame
        return f"<pre>{frame}</pre>"

    def build_accordion(self, outputs, directory):
        # return a lazy widgets.Accordion, or None to skip
        return None

sim = Simulation("my_run/")
sim.output_types.append(MyOutputType())
sim.outputs["my_id"][0].data   # calls MyOutputType.read_data
```

To apply a custom type to all future simulations, append to the class-level list:

```python
Simulation._default_output_type_classes.append(MyOutputType)
```

---

## Adding problem-specific routines

`Simulation` is assembled from two parts (see `__init__.py`):

```python
class Simulation(PhysicsMixin, BaseSimulation): ...
```

- `BaseSimulation` (`interface.py`) is the problem-agnostic core: config, history, outputs, display.
- `PhysicsMixin` (`physics.py`) holds problem-specific methods and derived quantities.

The mixin never imports the core; it just uses the core API through `self`
(`self.history`, `self.config`, `self.outputs`, ...). Since it comes first in the
method resolution order, it can both add methods and override core hooks.

To add a routine, add a method to `PhysicsMixin` in `physics.py`:

```python
class PhysicsMixin:
    def plot_my_quantity(self, ax=None):
        import matplotlib.pyplot as plt
        h = self.history
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(h["time"], h["KE"] / h["ME"])
        return ax
```

```python
sim.plot_my_quantity()
```

To change core behaviour without touching the core, override a hook. Currently
available: `_postprocess_history(df)` (called when building `sim.history`; must
return the DataFrame). The mixin uses it to add `v_rms` / `B_rms`:

```python
def _postprocess_history(self, df):
    df = super()._postprocess_history(df)
    df["my_column"] = ...
    return df
```

Keep anything that isn't problem-specific in the core. For a different problem
or code, write another mixin and combine it in `__init__.py`
(`class Simulation(MyMixin, BaseSimulation)`).

---

## Dependencies

| package | purpose |
|---------|---------|
| `numpy`, `pandas`, `matplotlib` | core analysis and plotting |
| `ipywidgets` | interactive notebook display |
| `openpmd-api` | reading OpenPMD/ADIOS2 `.bp` slice files |
| `yt` | reading Parthenon HDF5 `.phdf` snapshots |
