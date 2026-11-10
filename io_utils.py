import h5py
import pandas as pd
import re
import numpy as np
from collections import defaultdict
import os

# Suffixes energy_transfer::WriteResult (C++) appends to a spectrum's name for
# its three component records -- see energy_transfer/src/io_openpmd.cpp.
_ENERGY_TRANSFER_SPECTRUM_SUFFIXES = ("_pow_sum", "_k_sum", "_count_sum")

def extract_index(filename):
    """
    Extract the numeric index from a filename where the index is the last number
    before the file extension. Works for any extension.
    """
    filename_only = os.path.basename(filename)  # remove directory
    match = re.search(r'(\d+)\.[^.]+$', filename_only)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot parse index from filename: {filename}")

def read_cvs_hist(filename):
    df = pd.read_csv(filename, delim_whitespace=True, comment='#')
    # Extract column names from the first commented line
    with open(filename) as f:
        for line in f:
            if line.startswith("# ["):
                columns = [entry.split("=")[1].strip() for entry in line.split() if "=" in entry]
                break
    df.columns = columns
    # Compute dlog(ME)/dlog(t)
    df['dlogME_dlogt'] = np.gradient(np.log(df['ME']), np.log(df['time']))
    return df

def read_h5_to_dict(filename):
    def recursively_load(h5obj):
        data = {}
        for key, item in h5obj.items():
            if isinstance(item, h5py.Dataset):
                data[key] = item[()]  # load dataset into numpy array
            elif isinstance(item, h5py.Group):
                data[key] = recursively_load(item)
        return data
    
    with h5py.File(filename, 'r') as f:
        return recursively_load(f)

def parse_parthenon_input(filename):
    # Helper function to parse Parthenon input file
    data = defaultdict(dict)
    section = None

    with open(filename) as f:
        for line in f:
            # remove inline comments
            line = line.split("#", 1)[0].strip()
            # skip empty lines
            if not line:
                continue

            # check for section header
            m = re.match(r"<(.+?)>", line)
            if m:
                section = m.group(1).strip()
                if section not in data:
                    data[section] = {}
                continue

            # check for key = value
            if "=" in line and section is not None:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # convert numbers if possible
                try:
                    if "." in value or "e" in value.lower():
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # keep as string if not a number
                data[section][key] = value
    return dict(data)

def dict_to_array(transfer_dict):
    """
    Convert a nested dictionary of energy transfers into a 2D numpy array.

    Parameters
    ----------
    transfer_dict : dict
        Nested dictionary of the form dict[k_shell][Q_shell] = value

    Returns
    -------
    array : np.ndarray
        2D array with shape (num_k, num_Q)
    k_labels : list
        Ordered list of k shell labels
    Q_labels : list
        Ordered list of Q shell labels
    """
    # Outer keys = k shells (receiving)
    k_labels = sorted(transfer_dict.keys(), key=lambda x: float(x.split('-')[0]))
    # Inner keys = Q shells (giving)
    Q_labels = sorted(next(iter(transfer_dict.values())).keys(), key=lambda x: float(x.split('-')[0]))

    array = np.zeros((len(k_labels), len(Q_labels)))

    for i, k in enumerate(k_labels):
        for j, Q in enumerate(Q_labels):
            array[i, j] = transfer_dict[k][Q]

    return array, k_labels, Q_labels

def parse_spc_file(file_path):
    """
    Parses a whitespace-delimited file with columns:
    Bin, En_sum, K_sum, Count
    into a pandas DataFrame. Skips incomplete rows.
    """
    # Read the file, skipping empty lines
    df = pd.read_csv(
        file_path,
        sep=r'\s+',          # matches any whitespace
        comment='#',
        names=['Bin','En_sum','K_sum','Count'],
        skip_blank_lines=True
)    
    # Drop rows where any column is missing
    df = df.dropna()
    
    # Convert columns to appropriate types
    df['Bin'] = df['Bin'].astype(int)
    df['En_sum'] = df['En_sum'].astype(float)
    df['K_sum'] = df['K_sum'].astype(float)
    df['Count'] = df['Count'].astype(int)

    return df

def _is_energy_transfer_iteration(it):
    """
    True if openPMD iteration `it` was written by energy_transfer's C++
    WriteResult. WriteResult's shell-transfer and spectra parts are each
    optional, so either marker counts: the donor_shell_edges attribute
    (written with a shell-transfer result) or a "<name>_pow_sum" record
    (written per spectrum). An ordinary Parthenon openpmd output (e.g. a
    slice) shares the .bp extension but has neither.
    """
    if "donor_shell_edges" in it.attributes:
        return True
    return any(name.endswith(_ENERGY_TRANSFER_SPECTRUM_SUFFIXES[0]) for name in it.meshes)

def energy_transfer_file_info(path):
    """
    Lightweight check (metadata only, no mesh data read): is `path` an
    openPMD (.bp) series written by energy_transfer's C++ WriteResult (see
    _is_energy_transfer_iteration for how it's recognized)?

    Returns (is_energy_transfer_file: bool, time: float or None). Returns
    (False, None) for anything that isn't a readable openPMD series, rather
    than raising.
    """
    try:
        import openpmd_api as opmd
    except ImportError:
        return False, None
    try:
        s = opmd.Series(path, opmd.Access.read_only)
        it = s.iterations[next(iter(s.iterations))]
        is_et = _is_energy_transfer_iteration(it)
        time = it.time if is_et and it.time is not None else None
        s.close()
        return is_et, time
    except Exception:
        return False, None

def is_energy_transfer_file(path):
    """True if `path` is an energy_transfer .bp output -- see energy_transfer_file_info."""
    return energy_transfer_file_info(path)[0]

def parse_energy_transfer_file(path):
    """
    Parse a single-iteration openPMD (.bp) file written by energy_transfer's
    C++ WriteResult (see energy_transfer/src/io_openpmd.cpp) into plain
    Python data structures. Works standalone -- pass any path directly, no
    Simulation object needed.

    Parameters
    ----------
    path : str
        Path to the .bp series (a directory, for the BP5 engine).

    Returns
    -------
    dict with keys:
      "terms" : dict[str, np.ndarray]
          One array per computed term (e.g. "UUA", "BBA", ...), each shape
          (n_receiver, n_mediator, n_donor) -- receiver (K) outermost/rows,
          mediator in the middle, donor (Q) innermost/cols, matching this
          module's existing dict_to_array (K, Q) convention with the
          mediator axis sandwiched in. Any axis energy_transfer didn't
          resolve (collapsed to the whole domain) has length 1 -- squeeze()
          it out if not needed. Empty for a spectra-only output.
      "spectra" : dict[str, dict[str, np.ndarray]]
          One {"pow_sum": ..., "k_sum": ..., "count_sum": ...} per written
          spectrum (e.g. "spec_U"; a "<name>_decomp" request expands into
          "<name>", "<name>_compressive", "<name>_plus", "<name>_minus").
          Empty for a transfer-only output.
      "donor_edges" / "mediator_edges" / "receiver_edges" : np.ndarray or None
          Shell bin edges for each axis (length = that axis's shell count + 1).
          None for a spectra-only output (WriteResult writes them only
          alongside a shell-transfer result).
      "donor_binning" / "mediator_binning" / "receiver_binning" : str or None
          "lin", "log", or "custom"; None for a spectra-only output.
      "time" : float or None
          The iteration's simulation time, if the series has one.
    """
    try:
        import openpmd_api as opmd
    except ImportError:
        raise ImportError("openpmd-api not installed. Run: pip install openpmd-api")

    s = opmd.Series(path, opmd.Access.read_only)
    it_num = next(iter(s.iterations))
    it = s.iterations[it_num]

    if not _is_energy_transfer_iteration(it):
        s.close()
        raise ValueError(
            f"{path} does not look like an energy_transfer output "
            "(neither the donor_shell_edges attribute nor any *_pow_sum spectrum "
            "record that WriteResult writes)."
        )

    # Shell metadata is only written alongside a shell-transfer result.
    has_transfer = "donor_shell_edges" in it.attributes
    def attr(name, as_array=False):
        if not has_transfer:
            return None
        value = it.get_attribute(name)
        return np.array(value) if as_array else value

    donor_edges = attr("donor_shell_edges", as_array=True)
    mediator_edges = attr("mediator_shell_edges", as_array=True)
    receiver_edges = attr("receiver_shell_edges", as_array=True)
    donor_binning = attr("donor_binning")
    mediator_binning = attr("mediator_binning")
    receiver_binning = attr("receiver_binning")
    time = it.time if it.time is not None else None

    scalar = opmd.Mesh_Record_Component.SCALAR
    chunks = {mesh_name: it.meshes[mesh_name][scalar].load_chunk() for mesh_name in it.meshes}
    s.flush()

    terms = {}
    spectra = defaultdict(dict)
    for mesh_name, chunk in chunks.items():
        matched = next(
            (suf for suf in _ENERGY_TRANSFER_SPECTRUM_SUFFIXES if mesh_name.endswith(suf)), None
        )
        if matched is None:
            terms[mesh_name] = np.array(chunk)
        else:
            spectra[mesh_name[: -len(matched)]][matched[1:]] = np.array(chunk)
    s.close()

    return {
        "terms": terms,
        "spectra": dict(spectra),
        "donor_edges": donor_edges,
        "mediator_edges": mediator_edges,
        "receiver_edges": receiver_edges,
        "donor_binning": donor_binning,
        "mediator_binning": mediator_binning,
        "receiver_binning": receiver_binning,
        "time": time,
    }