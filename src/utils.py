# Utils for event_sampling
from box import Box
import cv2
import hdf5plugin
import h5py
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import numpy as np
import scipy
from scipy.spatial import KDTree
from scipy.stats import percentileofscore
import yaml


class EventFileSaver:
    def __init__(self, **kwargs):
        self.evs_write_path = kwargs.get("evs_write_path", None)  # Root path of the sequence
        self.dataset_type = kwargs.get("dataset_type", None)
        # Variables
        self.h5f = None
        self.h5f_init = False
        self.h5f_maxshape = kwargs.get("maxshape", None)
        self.h5_data_idx = 0
        self.divider = kwargs.get("divider", None)
        self.chunks = kwargs.get("chunks", None)
        self.size_y = kwargs.get("size_y", None)
        self.size_x = kwargs.get("size_x", None)
        # Init variables
        self._init_paths_()

    def _init_paths_(self):
        if self.evs_write_path is not None:
            self.evs_write_path.mkdir(parents=True, exist_ok=True)

    def _init_h5_(self, file_path):
        self.h5f = h5py.File(str(file_path), 'w')

        if self.dataset_type == 'esfp':
            # Original dataset structure:
            # p, dtype: uint8, shape: (109908,), maxshape: (109908,), compression: None, compression_opts: None, shuffle: False
            # t, dtype: int64, shape: (109908,), maxshape: (109908,), compression: None, compression_opts: None, shuffle: False
            # x, dtype: uint16, shape: (109908,), maxshape: (109908,), compression: None, compression_opts: None, shuffle: False
            # y, dtype: uint16, shape: (109908,), maxshape: (109908,), compression: None, compression_opts: None, shuffle: False

            self.h5f.create_dataset('p', dtype="<u1", shape=self.h5f_maxshape, maxshape=self.h5f_maxshape, chunks=self.chunks['p'], compression=32001, compression_opts=(0, 0, 0, 0, 1, 1, 5), shuffle=False)
            self.h5f.create_dataset('t', dtype="<i8", shape=self.h5f_maxshape, maxshape=self.h5f_maxshape, chunks=self.chunks['t'], compression=32001, compression_opts=(0, 0, 0, 0, 1, 1, 5), shuffle=False)
            self.h5f.create_dataset('x', dtype="<u2", shape=self.h5f_maxshape, maxshape=self.h5f_maxshape, chunks=self.chunks['x'], compression=32001, compression_opts=(0, 0, 0, 0, 1, 1, 5), shuffle=False)
            self.h5f.create_dataset('y', dtype="<u2", shape=self.h5f_maxshape, maxshape=self.h5f_maxshape, chunks=self.chunks['y'], compression=32001, compression_opts=(0, 0, 0, 0, 1, 1, 5), shuffle=False)

        else:
            # Original dataset structure:
            # events - [divider, height, p, t, width, x, y]
            # divider, <HDF5 dataset "divider": shape (), type "<i4">, int32, None
            # height, <HDF5 dataset "height": shape (), type "<i4">, int32, None
            # p, <HDF5 dataset "p": shape (822910,), type "|i1">, int8, (8192,)
            # t, <HDF5 dataset "t": shape (822910,), type "<i8">, int64, (2048,)
            # width, <HDF5 dataset "width": shape (), type "<i4">, int32, None
            # x, <HDF5 dataset "x": shape (822910,), type "<u2">, uint16, (4096,)
            # y, <HDF5 dataset "y": shape (822910,), type "<u2">, uint16, (4096,)

            # 1. Create events group
            ev_group = self.h5f.create_group('events')  # Key is "events", shape is for ()
            # 2. Create datasets for each element
            ev_group.create_dataset('divider', dtype="<i4", shape=(), data=self.divider)
            ev_group.create_dataset('height', dtype="<i4", shape=(), data=self.size_y)
            ev_group.create_dataset('width', dtype="<i4", shape=(), data=self.size_x)
            # Tried to match RVT compression
            ev_group.create_dataset('p', dtype="|i1", shape=8192, chunks=8192, maxshape=self.h5f_maxshape, compression=32001, compression_opts=(0, 0, 0, 0, 1, 1, 5), shuffle=False)
            ev_group.create_dataset('t', dtype="<i8", shape=2048, chunks=2048, maxshape=self.h5f_maxshape, compression=32001, compression_opts=(0, 0, 0, 0, 1, 1, 5), shuffle=False)
            ev_group.create_dataset('x', dtype="<u2", shape=4096, chunks=4096, maxshape=self.h5f_maxshape, compression=32001, compression_opts=(0, 0, 0, 0, 1, 1, 5), shuffle=False)
            ev_group.create_dataset('y', dtype="<u2", shape=4096, chunks=4096, maxshape=self.h5f_maxshape, compression=32001, compression_opts=(0, 0, 0, 0, 1, 1, 5), shuffle=False)

    def save_sampled_events(self, events_sampled, **kwargs):
        # Save sampled events
        if self.evs_write_path is not None:
            # Generate the file name (based on the .npz file) and write to disk
            evs_save_filename = kwargs.get('evs_filename', None)

            if evs_save_filename is not None:
                write_file_path = self.evs_write_path / evs_save_filename
                evs_transform = self.data_transform(events_sampled)
                self.write_ev_file(evs_transform, write_file_path)
        else:
            print("Warning: Events file is None") #TODO: Delete this

    def write_ev_file(self, evs_dict, file_path):
        # Write back the events file
        # For now do it Online
        # Save depending on extension
        if file_path.suffix == '.npz':
            np.savez(str(file_path), **evs_dict)
        elif file_path.suffix == '.npy':
            evs_x = evs_dict['x']
            evs_y = evs_dict['y']
            evs_t = evs_dict['t']
            evs_p = evs_dict['p']
            evs_np = np.stack((evs_x, evs_y, evs_t, evs_p), axis=1)
            np.save(str(file_path), evs_np)
        elif file_path.suffix == '.h5':
            if not self.h5f_init:
                self._init_h5_(file_path)
                self.h5f_init = True

            self.add_h5_data(evs_dict)  # Here has to be the original h5f data + events modified

    def data_transform(self, evs):
        # Transform event data to match dataset
        if self.dataset_type == 'caltech101rpg':
            # Caltech NPY files: (-1, 1). Do from (0,1) to (-1, 1)
            polarities = ((evs['p'] * 2) - 1)
            evs['p'] = polarities
            # Change data format to match Caltech / Cars datasets
            evs['x'] = evs['x'].astype(np.float32)
            evs['y'] = evs['y'].astype(np.float32)
            evs['t'] = evs['t'].astype(np.float32)
            evs['p'] = evs['p'].astype(np.float32)

        elif self.dataset_type == 'rvt':
            # Caltech NPY files: (-1, 1). Do from (0,1) to (-1, 1)
            polarities = ((evs['p'] * 2) - 1)
            evs['p'] = polarities

        return evs

    def add_h5_data(self, data):
        # Resize the dataset based on the incoming data. Probably not optimal.
        # Maybe generate the dataset's shape based on the acceptance probability
        array_len = len(data['t'])

        if array_len > 0:
            if self.dataset_type == 'esfp':
                new_size = self.h5_data_idx + array_len
                # Add new information
                self.h5f["x"].resize(new_size, axis=0)
                self.h5f["y"].resize(new_size, axis=0)
                self.h5f["t"].resize(new_size, axis=0)
                self.h5f["p"].resize(new_size, axis=0)

                self.h5f["x"][self.h5_data_idx:new_size] = data["x"]
                self.h5f["y"][self.h5_data_idx:new_size] = data["y"]
                self.h5f["t"][self.h5_data_idx:new_size] = data["t"]
                self.h5f["p"][self.h5_data_idx:new_size] = data["p"]

                self.h5_data_idx = new_size
            else:
                new_size = self.h5_data_idx + array_len
                # Add new information
                self.h5f["events"]["x"].resize(new_size, axis=0)
                self.h5f["events"]["y"].resize(new_size, axis=0)
                self.h5f["events"]["t"].resize(new_size, axis=0)
                self.h5f["events"]["p"].resize(new_size, axis=0)

                self.h5f["events"]["x"][self.h5_data_idx:new_size] = data["x"]
                self.h5f["events"]["y"][self.h5_data_idx:new_size] = data["y"]
                self.h5f["events"]["t"][self.h5_data_idx:new_size] = data["t"]
                self.h5f["events"]["p"][self.h5_data_idx:new_size] = data["p"]

                self.h5_data_idx = new_size


##+Event stream information+##

def get_seq_mevs(df_orig, dataset_type='caltech101'):
    # Sort the DataFrame by time (should be already sorted)
    df = df_orig.sort_values('t')
    # Get first and last event timestamp
    t_init = df_evs.iloc[0]['t']
    t_end = df_evs.iloc[-1]['t']
    # Get number of events
    n_evs = len(df.index)
    # Time difference
    if dataset_type == 'caltech101':
        d_time = t_end - t_init
    else:
        print(t_init)
        print(t_end) 
        print(t_end - t_init)
        d_time = (t_end - t_init) * 1e-6  # Timelens in microseconds
        
    mevs = (n_evs / d_time) * 1e-6  # MegaEvents per second
    return mevs, n_evs, d_time

##-Event stream information-##


##+ YAML +##
def expand_envs(obj):
    if isinstance(obj, dict):
        return {k: expand_envs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_envs(v) for v in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
        config_dict = expand_envs(config_dict)  # Expand environment variables
    return Box(config_dict)  # Convert dict to Box for dot notation access


def print_recursive(config, indent=0):
    """Recursively print dictionary or Box object with indentation."""
    for key, value in config.items():
        if isinstance(value, dict):  # If the value is a dictionary (or Box), recurse
            print(" " * indent + f"{key}:")
            print_recursive(value, indent + 4)  # Indent nested keys
        else:
            print(" " * indent + f"{key}: {value}")

def get_cfg_sampling_paths(cfg):
    """Get the paths from the config."""
    paths_dict = {
        'seqs_path': None,
        'evs_path': None,
        'pcd_path': None,
        'img_path': None,
        'tstmp_path': None,
        'evs_save_path': None,
        'prior_path': None,
    }
    # Read the events from the input file, assigning header to columns
    if cfg.dataset.name == 'EventCameraDataset':
        # Establish the input and output paths
        paths_dict['evs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_filename}")
        paths_dict['pcd_path'] = Path(f"{cfg.path.pcd_root}/{cfg.path.evs_filename}").with_suffix('.pcd')

    elif cfg.dataset.name == 'NCaltech101':
        # Establish the input and output paths
        paths_dict['seqs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.dataset.split}")
        paths_dict['evs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.dataset.split}")
        paths_dict['pcd_path'] = Path(f"{cfg.path.root}/{cfg.path.pcd}/{cfg.path.evs_sampling}/{cfg.dataset.split}")
        paths_dict['evs_save_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_save}/{cfg.sampler_params.exp_name}")

    elif cfg.dataset.name == 'hsergb':
        # Establish the input and output paths
        paths_dict['seqs_path'] = Path(f"{cfg.path.root}")
        paths_dict['evs_path'] = Path(f"{cfg.path.root}")
        paths_dict['img_path'] = Path(f"{cfg.path.root}/{cfg.path.seq}/{cfg.path.img}")
        paths_dict['tstmp_path'] = Path(f"{cfg.path.root}/{cfg.path.seq}/{cfg.path.tstamp}")
        paths_dict['evs_save_path'] = Path(f"{cfg.path.root}")

    elif cfg.dataset.name == 'gen1':
        # Establish the input and output paths
        paths_dict['seqs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.path.evs}/{cfg.dataset.split}")
        paths_dict['evs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.path.evs}/{cfg.dataset.split}")
        paths_dict['pcd_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.path.pcd}/{cfg.dataset.split}")
        paths_dict['evs_save_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_save}/{cfg.sampler_params.exp_name}")
        paths_dict['prior_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.path.evs}/{cfg.path.prior}/{cfg.path.prior_file}")

    elif cfg.dataset.name == 'esfp':
        # Establish the input and output paths
        paths_dict['seqs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.dataset.split}")
        paths_dict['evs_save_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_save}/{cfg.sampler_params.exp_name}")
        
    return paths_dict

def get_cfg_paths(cfg):
    # For debugging
    """Get the paths from the config."""
    paths_dict = {
        'seqs_path': None,
        'evs_path': None,
        'pcd_path': None,
        'img_path': None,
        'tstmp_path': None,
        'evs_save_path': None,
        'prior_path': None,
    }

    # Read the events from the input file, assigning header to columns
    if cfg.dataset.name == 'EventCameraDataset':
        # Establish the input and output paths
        paths_dict['evs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_filename}")
        paths_dict['pcd_path'] = Path(f"{cfg.path.pcd_root}/{cfg.path.evs_filename}").with_suffix('.pcd')

    elif cfg.dataset.name == 'NCaltech101':
        # Establish the input and output paths
        paths_dict['seqs_path'] = ""
        paths_dict['evs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.dataset.split}/{cfg.path.label}/{cfg.path.seq}")
        paths_dict['pcd_path'] = Path(f"{cfg.path.root}/{cfg.path.pcd}/{cfg.path.evs_sampling}/{cfg.dataset.split}/{cfg.path.label}/{cfg.path.seq}").with_suffix('.pcd')
        paths_dict['evs_save_path'] = Path(f"/home/user/app/tmp/{cfg.dataset.name}/{cfg.path.label}")

    elif cfg.dataset.name == 'hsergb':
        # Establish the input and output paths
        paths_dict['seqs_path'] = Path(f"{cfg.path.root}")
        paths_dict['evs_path'] = Path(f"{cfg.path.root}/{cfg.path.seq}/{cfg.path.evs}/{cfg.path.evs_sampling}")
        paths_dict['pcd_path'] = Path(f"{cfg.path.root}/{cfg.path.seq}/{cfg.path.pcd}/{cfg.path.evs_sampling}/{cfg.path.seq}").with_suffix('.pcd')
        paths_dict['img_path'] = Path(f"{cfg.path.root}/{cfg.path.seq}/{cfg.path.img}")
        paths_dict['tstmp_path'] = Path(f"{cfg.path.root}/{cfg.path.seq}/{cfg.path.tstamp}")
        paths_dict['evs_save_path'] = Path(f"/home/user/app/tmp/{cfg.dataset.name}")

    elif cfg.dataset.name == 'gen1':
        # Establish the input and output paths
        paths_dict['seqs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.path.evs}/{cfg.dataset.split}")
        paths_dict['evs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.path.evs}/{cfg.dataset.split}/{cfg.path.seq}_td.dat.h5")
        paths_dict['pcd_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.path.pcd}/{cfg.dataset.split}/{cfg.path.seq}").with_suffix('.pcd')
        paths_dict['evs_save_path'] = Path(f"/home/user/app/tmp/{cfg.dataset.name}/{cfg.path.seq}")
        paths_dict['prior_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.path.evs}/{cfg.path.prior}/{cfg.path.prior_file}")

    elif cfg.dataset.name == 'esfp':
        # Establish the input and output paths
        paths_dict['seqs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.dataset.split}")
        paths_dict['evs_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.dataset.split}/{cfg.path.seq}/events.h5")
        paths_dict['pcd_path'] = Path(f"{cfg.path.root}/{cfg.path.evs_sampling}/{cfg.dataset.split}/{cfg.path.seq}/events.pcd")
        paths_dict['evs_save_path'] = Path(f"/home/user/app/tmp/{cfg.dataset.name}/{cfg.dataset.split}")

    return paths_dict

##- YAML -##

##+ Config +##
def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries.
    - If the same key exists in both dictionaries and values are dictionaries, merge them.
    - Otherwise, the value from dict2 overrides the value from dict1.
    """
    merged = dict1.copy()  # Start with a shallow copy of dict1
    for key, value in dict2.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # If both are dicts, merge recursively
            merged[key] = merge_dicts(merged[key], value)
        else:
            # Otherwise, override or add
            merged[key] = value
    return Box(merged)

def flatten_dict(d):
    """
    Flatten a dictionary by discarding parent keys and only keeping the last level keys.
    """
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value))  # Recursively flatten nested dicts
        else:
            flat_dict[key] = value
    return flat_dict

def merge_configs(default_cfg, cli_args):
    """
    Merge default config and CLI args into a flat dictionary with single-level keys.
    """
    merged_cfg = Box(default_cfg)  # Start with defaults
    for key, value in vars(cli_args).items():
        if value is not None:  # Only override if CLI argument is provided
            merged_cfg[key] = value

    # Flatten the merged configuration
    return flatten_dict(merged_cfg.to_dict())

##- Config -##


##+ Prior +##
#def load_masks_json(json_path, labels=["phone"]):
def load_masks_json(json_path, labels=["object"]):
    # Load the masks from a JSON file

    # Labelme JSON format
    # {
    #     "version": "4.5.12",
    #     "flags": {},
    #     "shapes": [
    #         {
    #         "label": "phone",
    #         "points": [
    #             [
    #             416.73913043478257,
    #             288.0869565217391
    #             ],
    #             [
    #             511.3043478260869,
    #             485.9130434782609
    #             ]
    #         ],
    #         "group_id": null,
    #         "shape_type": "rectangle",
    #         "flags": {}
    #         }
    #     ],
    #     "imagePath": "bowl1_31_10.png",
    #     "imageData": null,
    #     "imageHeight": 720,
    #     "imageWidth": 1280
    # }

    with open(json_path, "r") as f:
        data = json.load(f)

    # Get image dimensions
    height = data["imageHeight"]
    width = data["imageWidth"]

    # Create an empty binary mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Loop through the shapes and find the "phone" label
    for shape in data["shapes"]:
        label = shape["label"]
        if label in labels:
            (x1, y1), (x2, y2) = shape["points"]  # Get rectangle coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=1, thickness=-1)  # Draw white rectangle

    return mask


##- Prior -##


##+ Event preprocessing +##
def filter_events_to_img(img_size, x, y, t, p):
    # Filter events outside the image bounds
    img_h = img_size[0]
    img_w = img_size[1]
    mask = (x <= img_w - 1) & (y <= img_h - 1) & (x >= 0) & (y >= 0)
    x_ = x[mask]
    y_ = y[mask]
    t_ = t[mask]
    p_ = p[mask]

    return x_, y_, t_, p_


def normalize_events(events, img_size, twindow_size):
    # Normalize the events to the image size
    x = events['x']
    y = events['y']
    t = events['t']
    p = events['p']

    #x_, y_, t_, p_ = filter_events_to_img(img_size, x, y, t, p)

    # Normalize the events
    x_norm = x / img_size[1]
    y_norm = y / img_size[0]
    t_norm = t / twindow_size
    p_norm = p

    evs_norm = {'x': x_norm, 'y': y_norm, 't': t_norm, 'p': p_norm}

    return evs_norm


def unnormalize_events(events, img_size, twindow_size):
    # Unnormalize the events to the image size
    x = events['x']
    y = events['y']
    t = events['t']
    p = events['p']

    x_unnorm = x * img_size[1]
    y_unnorm = y * img_size[0]
    t_unnorm = t * twindow_size
    p_unnorm = p

    evs_unnorm = {'x': x_unnorm, 'y': y_unnorm, 't': t_unnorm, 'p': p_unnorm}

    return evs_unnorm


def voxelize_events(events, img_size, twindow_size, voxel_size_spat, voxel_size_temp):
    # Compute tdiffs per voxel
    events_tdiff = compute_evs_tdiff(events)

    # Voxelize the events to a 3D grid defined by the voxel sizes (in pixels)
    x = events['x']
    y = events['y']
    t = events['t']
    p = events['p']
    t_diff = events_tdiff['delta_t']
    
    # Voxelize the events
    # 0..voxel_size_spat-1, voxel_size_spat..2*voxel_size_spat-1, ...
    x_vox = (x // voxel_size_spat).astype(int)
    y_vox = (y // voxel_size_spat).astype(int)
    t_vox = (t // voxel_size_temp).astype(int)
    

    # Create a 3D histogram of the voxelized events
    evs_vox, _, _ = np.histogram2d(x_vox, y_vox, bins=(img_size[1] // voxel_size_spat, img_size[0] // voxel_size_spat))

    # Histogram of time differences
    evs_tdiff_vox, _, _ = np.histogram(t_vox, bins=(twindow_size // voxel_size_temp))

    # Histogram of time differences 2
    evs_tdiff_vox2, _, _ = np.histogram(t_diff_vox, bins=(twindow_size // voxel_size_temp))

    # Plot the voxelized events
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(evs_vox.T, origin='lower', cmap='gray')
    plt.title('Voxelized Events')
    plt.xlabel('X (Voxels)')
    plt.ylabel('Y (Voxels)')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(evs_tdiff_vox)), evs_tdiff_vox, color='gray')
    plt.title('Voxelized Time Differences')
    plt.xlabel('Time (Voxels)')
    plt.ylabel('Events')
    plt.tight_layout()
    plt.show()
   
    return evs_vox, evs_tdiff_vox


def assign_events_to_grid(df, grid_shape, spatial_dims, temporal_range):
    """
    Assign events to a 3D grid.

    Parameters:
    - df: Pandas DataFrame with columns ['x', 'y', 't', 'p'].
    - grid_shape: Tuple (x_bins, y_bins, t_bins) defining the number of bins in each dimension.
    - spatial_dims: Tuple (width, height) of the sensor dimensions in pixels.
    - temporal_range: Tuple (t_min, t_max) defining the temporal range in seconds.

    Returns:
    - A dataframe with the bin information per event
    """
    # Unpack grid shape and spatial/temporal ranges
    x_bins, y_bins, t_bins = grid_shape
    width, height = spatial_dims
    t_min, t_max = temporal_range

    # Define bin edges
    x_edges = np.linspace(0, width, x_bins + 1)
    y_edges = np.linspace(0, height, y_bins + 1)
    t_edges = np.linspace(t_min, t_max, t_bins + 1)

    # Assign events to bins
    df['x_bin'] = np.digitize(df['x'], bins=x_edges) - 1
    df['y_bin'] = np.digitize(df['y'], bins=y_edges) - 1
    df['t_bin'] = np.digitize(df['t'], bins=t_edges) - 1
    
    return df


# Function to return the df_evs partitioned in a time window
def partition_df_tw(df, t_window, **kwargs):
    t_start = kwargs.get('t_start', df['t'].min())
    t_end = kwargs.get('t_end', df['t'].max())
    tdiff_start = kwargs.get('tdiff_start', 0)
    tdiff_end = kwargs.get('tdiff_end', t_window)  # max observable tdiff inside twindow

    t_start_ = t_start
    t_end_ = t_start + t_window

    df_tw_tdiff_list = []
    id_tw_tdiff_list = []

    # No overlapping
    while True:
        df_original_diff_tw = generate_df_tdiff(df, t_start_, t_end_, tdiff_start, tdiff_end)
        df_tw_tdiff_list.append(df_original_diff_tw)
        id_tw_tdiff_list.append(t_start_)
    
        t_start_ += t_window
        t_end_ += t_window
    
        if t_start_ >= t_end:
            break

    return df_tw_tdiff_list, id_tw_tdiff_list


# Filter events in bboxes (left, top, w, h)
def filter_events_in_bboxes(df_evs, df_bbox_data):
    t_steps_frames = df_bbox_data['t'].unique()
    
    # Build time-bbox tuples
    t_bbox_list = []
    for t_step_frame in t_steps_frames:
        mask_t = df_bbox_data['t'] == t_step_frame
        t_bbox_list.append((t_step_frame, df_bbox_data[mask_t]))
    
    df_evs_in_bboxes_list = []
    t_step_frame_prev = 0
    for t_step_frame, df_step_bboxes in t_bbox_list:
        mask_t = (df_evs['t'] >= t_step_frame_prev) & (df_evs['t'] < t_step_frame)
        df_evs_t_ = df_evs[mask_t]
    
        for _, row in df_step_bboxes.iterrows():
            mask_sp = (df_evs_t_['x'] >= row['x']) & (df_evs_t_['y'] >= row['y']) & (df_evs_t_['x'] <= (row['x'] + row['w'])) & (df_evs_t_['y'] <= (row['y'] + row['h']))
            df_evs_in_bboxes_list.append(df_evs_t_[mask_sp])

        t_step_frame_prev = t_step_frame
    
    df_evs_in_bboxes = pd.concat(df_evs_in_bboxes_list).reset_index()
    
    return df_evs_in_bboxes


def df_list_to_hist_list(df_list, n_bins=100):
    # Include min-max tdiff for histogram alignment
    min_tdiff, max_tdiff = get_min_max_tdiff(df_list)

    # Generate the histogram of the dataframes in the list
    n_bins = 100
    
    hist_bins_list = []
    for df in df_list:
        df_ = add_rows_tdiff(df, min_tdiff, max_tdiff)
        hist_ = compute_tdiff_pdf(df_, bins=n_bins)
        hist_[0][np.isnan(hist_[0])] = 0
        hist_bins_list.append(hist_)

    return hist_bins_list

##- Event preprocessing -##


##+ Event sampling +##

def vectorized_ber(scores):
    # Vectorized random selection based on probabilities
    #is_accept_ber = np.random.rand(len(scores)) < scores  # Vectorized Bernoulli sampling. Equivalent vectorized form to random.choices([True, False], probabilities). E.g. if 0.5 < 0.9 -> True
    # Test, accept everything (sampling is done afterwards)
    is_accept_ber = np.ones(len(scores), dtype=bool)

    return is_accept_ber


def sample_events_to_df(evs_array, scores, seqs_len, sampling_rate=0.5):   
    # Vectorized random selection based on probabilities
    is_accept_ber = vectorized_ber(scores)

    # Take the accepted events (is_accept_ber), reorder them and keep the top N events
    accepted_scores = scores[is_accept_ber]
    accepted_idx = np.argsort(accepted_scores)[::-1]  # Return the indices that would sort the array in descending order (max to min)
    accepted_idx = accepted_idx[:max(1, int(sampling_rate * seqs_len))]  # Keep the top N events (accept at least 1)
    accepted_idx = np.sort(accepted_idx)  # Sort the indices

    # Create a boolean mask for the accepted events
    is_accept = np.zeros(seqs_len, dtype=bool)
    is_accept[accepted_idx] = True

    # Get the coordinates and timestamps of the accepted events
    x_coords = evs_array[0][accepted_idx]
    y_coords = evs_array[1][accepted_idx]
    t_vals   = evs_array[2][accepted_idx]
    p_vals   = evs_array[3][accepted_idx]

    # Gather accepted events
    downsampled_evs = np.column_stack([x_coords, y_coords, t_vals, p_vals])
    df_downsampled_evs = pd.DataFrame(downsampled_evs, columns=['x', 'y', 't', 'p'])

    # Sort the events by timestamp
    df_downsampled_evs = df_downsampled_evs.sort_values('t')    

    return df_downsampled_evs, is_accept, is_accept_ber

##- Event sampling -##

##+ PointCloud +##
import torch

def evs_to_pointcloud(evs, tscale=1e6, device='cpu'):
    points = torch.tensor(evs[['x', 'y', 't']].values, dtype=torch.float32)
    points[:, 2] = points[:, 2] * tscale
    points.to(device)

    return points


def chamfer_distance_simple_onesided(P, Q):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(Q)
    dist_P = tree.query(P)[0]
    
    return np.mean(dist_P)


def chamfer_distance_vanilla(P, Q):
    """
    Compute the Chamfer Distance between two point clouds.
    Args:
        P: Tensor of shape (batch_size, num_points, 3)
        Q: Tensor of shape (batch_size, num_points, 3)
    Returns:
        chamfer_loss: Scalar value of the Chamfer Distance between the point clouds
    """
    # Compute pairwise distance
    diff = P.unsqueeze(2) - Q.unsqueeze(1)  # Shape: (batch_size, num_points1, num_points2, 3)
    dist = torch.sum(diff ** 2, dim=-1)  # Shape: (batch_size, num_points1, num_points2)

    # For each point in P, find the closest point in Q
    min_dist1, _ = torch.min(dist, dim=2)  # Shape: (batch_size, num_points1)
    # For each point in Q, find the closest point in P
    min_dist2, _ = torch.min(dist, dim=1)  # Shape: (batch_size, num_points2)

    # Compute the Chamfer loss
    chamfer_loss = torch.mean(min_dist1) + torch.mean(min_dist2)
    
    return chamfer_loss


def remove_exact_matches(P, Q):
    """
    Remove points from P that are exactly the same as points in Q.
    Args:
        P: Tensor of shape (num_points_P, 3) representing the complete point cloud.
        Q: Tensor of shape (num_points_Q, 3) representing the downsampled point cloud.
    Returns:
        reduced_P: Tensor with points removed that match points in Q.
    """
    # Expand P and Q to have shapes compatible for broadcasting
    P_expanded = P.unsqueeze(1)  # Shape: (num_points_P, 1, 3)
    Q_expanded = Q.unsqueeze(0)  # Shape: (1, num_points_Q, 3)

    # Check for exact matches
    matches = (P_expanded == Q_expanded).all(dim=2)  # Shape: (num_points_P, num_points_Q)

    # Any point in P that matches any point in Q will have a "True" value
    matched_points = matches.any(dim=1)  # Shape: (num_points_P)

    # Invert the mask to keep points in P that do not match points in Q
    reduced_P = P[~matched_points]

    return reduced_P


def chamfer_distance_one_sided(P, Q):
    """
    Compute the one-sided Chamfer Distance from P to Q, assuming Q is a subset of P.
    Args:
        P: Tensor of shape (num_points_P, 3).
        Q: Tensor of shape (num_points_Q, 3).
    Returns:
        chamfer_loss: Scalar value of the one-sided Chamfer Distance between P and Q.
    """
    # Remove exact matches from P
    reduced_P = remove_exact_matches(P, Q)  # Reduces 4 seconds the computation
    # Time elapsed for remove_exact_matches: 0.7733452320098877

    # Compute pairwise distances from points in reduced_P to points in Q
    diff = reduced_P.unsqueeze(1) - Q.unsqueeze(0)  # Shape: (num_points_reduced_P, num_points_Q, 3)
    # Time elapsed for diff: 0.29576635360717773
    dist = torch.sum(diff ** 2, dim=-1)  # Shape: (num_points_reduced_P, num_points_Q)
    # Time elapsed for dist: 0.925994873046875

    # For each point in reduced_P, find the closest point in Q
    min_dist, _ = torch.min(dist, dim=1)  # Shape: (num_points_reduced_P)

    # Compute the one-sided Chamfer loss
    chamfer_loss = torch.sum(min_dist) / P.shape[0]
    
    return chamfer_loss


##- PointCloud -##


def tranform_pdf_to_percentile_rank(pdf):
    # Flatten and sort the PDF values
    flattened_pdf = pdf.flatten()
    sorted_pdf_values = np.sort(flattened_pdf)

    # Precompute percentile ranks for each position
    percentile_ranks = np.zeros_like(pdf, dtype=float)

    for x in range(pdf.shape[0]):
        for y in range(pdf.shape[1]):
            new_sample_likelihood = pdf[x, y]
            percentile_ranks[x, y] = percentileofscore(sorted_pdf_values, new_sample_likelihood)

    # # Now, whenever you have a new sample's position [x_new, y_new], you can look up the precomputed percentile rank
    # new_sample_position = [5, 8]  # Replace with your sample's position
    # precomputed_percentile_rank = percentile_ranks[new_sample_position[0], new_sample_position[1]]

    return percentile_ranks


def get_percentile_rank(sample, pdf, sorted_pdf_values=None):
    if sorted_pdf_values is None:
        # Flatten and sort the PDF values
        flattened_pdf = pdf.flatten()
        sorted_pdf_values = np.sort(flattened_pdf)

    # sample = [x, y]
    sample_likelihood = pdf[sample]

    return percentileofscore(sorted_pdf_values, sample_likelihood)


def tranform_pdf_to_percentile_rank_nearest(pdf):
    # compute values corresponding to the indicated percentiles
    percentiles = np.percentile(pdf, q=[0, 10, 25, 50, 75, 90])

    # Precompute nearest percentile ranks for each position
    percentile_ranks = np.zeros_like(pdf, dtype=float)

    for x in range(pdf.shape[0]):
        for y in range(pdf.shape[1]):
            new_sample_likelihood = pdf[x, y]
            percentile_ranks[x, y] = percentiles[np.argmin(np.abs(percentiles - new_sample_likelihood))]

    return percentile_ranks


def count_decimal_positions(number):
    # Convert the number to a string
    number_str = str(number)

    # Check if the number has a decimal point
    if '.' in number_str:
        # Find the position of the decimal point
        decimal_position = number_str.index('.')

        # Count the number of characters after the decimal point
        decimal_count = len(number_str) - decimal_position - 1
        return decimal_count
    else:
        # If the number is an integer (has no decimal point)
        return 0


def generate_gaussian_prior(size_y, size_x, std_y=1, std_x=1):
    # Parameters for the Gaussian distribution
    mean_y, mean_x = 0, 0  # Mean (center of the distribution)
    sigma_y, sigma_x = std_y, std_x  # Standard deviation

    # Create a grid of (x, y) coordinates
    x = np.linspace(-1, 1, size_x)
    y = np.linspace(-1, 1, size_y)
    x, y = np.meshgrid(x, y)

    # Apply the 2D Gaussian formula
    pdf_array = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
                -((x - mean_x)**2 / (2 * sigma_x**2) + (y - mean_y)**2 / (2 * sigma_y**2))
                )

    return pdf_array


def load_prior(prior_path):
    return np.load(prior_path)


def fit_plane(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points, rowvar=False)
    _, _, vh = np.linalg.svd(cov_matrix)
    normal_vector = vh[-1]
    if normal_vector[2] < 0:
        normal_vector = -normal_vector
    return normal_vector, centroid


def compute_distances(points, normal_vector, point_on_plane):
    distances = np.abs((points - point_on_plane) @ normal_vector)
    return distances


####+ Events tdiff +####
# Function per dataframe
def generate_df_tdiff(df, t_start, t_end, tdiff_start=0, tdiff_end=np.inf, **kwargs):
    area_size = kwargs.get('area_size', 1)

    # Filter and generate df_tdiff
    df_ = filter_evs(df, tstart=t_start, tend=t_end)
    df_ = compute_evs_tdiff_area(df_, area_size=area_size)
    df_ = filter_evs_tdiff(df_, xstart=tdiff_start, xend=tdiff_end)

    return df_


# Compute min and max tdiff to align histograms
def get_min_max_tdiff(df_list):
    min_tdiff = 1e8
    max_tdiff = 0

    for df_ in df_list:
        min_tdiff_ = df_['delta_t'].min()
        max_tdiff_ = df_['delta_t'].max()

        min_tdiff = min_tdiff_ if min_tdiff_ < min_tdiff else min_tdiff
        max_tdiff = max_tdiff_ if max_tdiff_ > max_tdiff else max_tdiff

    return min_tdiff, max_tdiff


def compute_evs_tdiff_with_polarity_change(df_evs):
    df_evs_tdiff = df_evs.copy()
    df_evs_tdiff['delta_t'] = df_evs_tdiff.groupby(['x', 'y'])['t'].diff()
    df_evs_tdiff['polarity_change'] = (df_evs_tdiff['p'] != df_evs_tdiff['p'].shift()).astype(int)
    df_evs_tdiff['delta_t'] = df_evs_tdiff.apply(lambda row: -1 if row['polarity_change'] == 0 else row['delta_t'], axis=1)
    return df_evs_tdiff


def compute_evs_tdiff(df_evs):
    df_evs_res = df_evs.copy()
    df_evs_res = df_evs_res.sort_values('t')
    df_evs_res['delta_t'] = df_evs.groupby(['x', 'y'])['t'].diff()
    return df_evs_res


def compute_evs_tdiff_area(df_evs, area_size=1):
    df_evs_res = df_evs.copy()
    df_evs_res = df_evs_res.sort_values('t')
    df_evs_res['delta_t'] = df_evs.groupby([df_evs['x'] // area_size, df_evs['y'] // area_size])['t'].diff()
    #if area_size > 1:
    #    df_evs_res['delta_t'] = df_evs_res.groupby([df_evs_res['x'] // area_size, df_evs_res['y'] // area_size])['delta_t'].apply(lambda x: x.diff())
    return df_evs_res


def filter_evs(df_evs, tstart=0, tend=1e8):
    return df_evs[(df_evs['t'] >= tstart) & (df_evs['t'] <= tend)]


def filter_evs_tdiff(df_evs, xstart=0, xend=1e8):
    # Also removes the nans (single event)
    return df_evs[(df_evs['delta_t'] >= xstart) & (df_evs['delta_t'] <= xend)]


def add_rows_tdiff(df_evs, min_dt=0, max_dt=1):
    # df_evs['delta_t'] = df_evs.groupby(['x', 'y'])['t'].diff()  # --> Maybe a typo?
    # Add a row for delta_t = 0 in position -1, -1 and t=-1 at the beginning of the dataframe
    new_row_first = pd.DataFrame({'x': [-1], 'y': [-1], 't': [-1], 'p': [1], 'delta_t': [min_dt]})
    new_row_last = pd.DataFrame({'x': [-1], 'y': [-1], 't': [-1], 'p': [1], 'delta_t': [max_dt]})
    df_evs = pd.concat([new_row_first, df_evs, new_row_last], ignore_index=True)
    return df_evs


def compute_tdiff_hist(df_evs, bins=100):
    hist, bins = np.histogram(df_evs['delta_t'], bins=bins)
    return hist, bins


def compute_tdiff_pdf(df_evs, bins=100):
    hist, bins = np.histogram(df_evs['delta_t'].dropna(), bins=bins, density=True)
    hist_pdf = hist / hist.sum()
    return hist_pdf, bins


def plot_evs_tdiff(df_evs, xstart=0, xend=1e8):
    # Test: Check events interval per pixel
    # 1. Check the interval of e1 and e2, e2 and e3... pixel by pixel
    # 2. Make 2 plots, one 1D (to see the global distribution / modes) and one 2D (same but to compare with prior)
    # Group by 'x' and 'y' columns and compute the difference
    df_evs = compute_evs_tdiff(df_evs)
    df = filter_evs_tdiff(df_evs, xstart, xend)

    # Create histogram from Pandas
    hist = df['delta_t'].hist(bins=50, alpha=0.75)
    hist.set_title('Delta t Differences Event Distribution')
    hist.set_xlabel('Delta t (us)')
    hist.set_ylabel('Events')
    hist.grid(True)

    plt.show()

    return hist


def evs_to_tdiff_pdf(evs, area_size=2, bins=20):
    """Compute the PDF of the time difference between events in a 2x2 area."""
    df_evs = pd.DataFrame(evs)
    df_evs_diff = compute_evs_tdiff_area(df_evs, area_size=area_size)
    pdf, bins = compute_tdiff_pdf(df_evs_diff, bins=bins)

    return pdf, bins


def df_evs_to_tdiff_pdf(df_evs, area_size=2, bins=20):
    """Compute the PDF of the time difference between events in a 2x2 area."""
    df_evs_diff = compute_evs_tdiff_area(df_evs, area_size=area_size)
    pdf, bins = compute_tdiff_pdf(df_evs_diff, bins=bins)

    return pdf, bins


######

def df_to_mat(df_evs, t_res=0.01):
    # Compute the SVD parting from an event dataframe.
    # t_res indicates the temporal resolution, so events' time will be fit to such a resolution (e.g. 0.045 will be mapped to 0.05)
    # Get the max x, y, and t values
    max_x = int(df_evs['x'].max()) + 1
    max_y = int(df_evs['y'].max()) + 1
    max_t = int(df_evs['t'].max() / t_res) + 1
    
    # Initialize the matrices
    mat_pol = np.zeros((max_x, max_y, max_t))
    mat_hasev = np.zeros((max_x, max_y, max_t))
    
    # Populate the matrices
    for index, row in df_evs.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        t = int(row['t'] / t_res)
        mat_pol[x, y, t] = row['p']  # Last event will be the one remaining
        mat_hasev[x, y, t] = 1
    
    return mat_pol, mat_hasev


def df_to_mat(df_evs, t_res=0.01):
    # Compute the SVD parting from an event dataframe.
    # t_res indicates the temporal resolution, so events' time will be fit to such a resolution (e.g. 0.045 will be mapped to 0.05)
    # Get the max x, y, and t values
    max_x = int(df_evs['x'].max()) + 1
    max_y = int(df_evs['y'].max()) + 1
    max_t = int(df_evs['t'].max() / t_res) + 1
    
    # Initialize the matrices
    mat_pol = np.zeros((max_x, max_y, max_t))
    mat_hasev = np.zeros((max_x, max_y, max_t))
    mat_delta_t = np.zeros((max_x, max_y, max_t))
    mat_last_t = np.zeros((max_x, max_y))
    
    # Populate the matrices
    for index, row in df_evs.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        t = int(row['t'] / t_res)
        mat_pol[x, y, t] = row['p']  # Last event will be the one remaining
        mat_hasev[x, y, t] = 1
        
        # Calculate delta_t for the current event
        delta_t = row['t'] - mat_last_t[x, y]
        mat_delta_t[x, y, t] = delta_t

        # Add last time
        mat_last_t[x, y] = row['t']

    
    return mat_pol, mat_hasev, mat_delta_t


def compute_svd(df_evs, t_res=0.01):
    # dataframe to 3D matrix
    mat_pol, math_hasev = df_to_mat(df_evs)

    # Step 1: Reshape the 3D matrix (100x100x10) to 2D (10000x10)
    A_flat = math_hasev.reshape(math_hasev.shape[0] * math_hasev.shape[1], math_hasev.shape[2])

    # Step 2: Perform SVD on the reshaped matrix
    U, S, Vt = np.linalg.svd(A_flat, full_matrices=False)

    return U, S, Vt
    