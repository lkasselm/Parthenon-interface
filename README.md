# Simulation Interface

A Jupyter notebook interface for exploring [AthenaPK](https://github.com/parthenon-hpc-lab/athenapk) / [Parthenon](https://github.com/parthenon-hpc-lab/parthenon) simulation output.

## Setup

```python
import sys
sys.path.insert(0, "/path/to/interface")
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
df = sim.history   # pandas DataFrame, columns = all history quantities + "time"
df["ME"]           # magnetic energy time series
```

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

## Dependencies

| package | purpose |
|---------|---------|
| `numpy`, `pandas`, `matplotlib` | core analysis and plotting |
| `ipywidgets` | interactive notebook display |
| `openpmd-api` | reading OpenPMD/ADIOS2 `.bp` slice files |
| `yt` | reading Parthenon HDF5 `.phdf` snapshots |
