import h5py 
import pandas as pd 
import re 
import numpy as np 
from collections import defaultdict
import os

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