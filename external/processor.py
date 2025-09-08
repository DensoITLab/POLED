# Copyright: Andreu Girbau-Xalabarder, 2025
# We provide the code as-is,
# without warranty of any kind, express or implied.

"""
Event Processing Library

This library provides a set of classes for processing events from various event-based sensors and datasets.

Usage:
1. Import the appropriate EventProcessor subclass for your dataset.
2. Initialize the processor with the necessary parameters (e.g., file paths, bin size).
3. Use methods like get_next_bins() or read_events_until() to process events.

Example:
    from processor import EventProcessorVid2e

    processor = EventProcessorVid2e(evs_path='path/to/events', imgs_path='path/to/images', bin_size=33)
    events = processor.get_next_bins(num_bins=1)
    # Process events...

Available EventProcessor subclasses:
- EventProcessorVid2e
- EventProcessorDVSVoltmeter
- EventProcessorTimelens
- EventProcessorRVT
- EventProcessorCaltech101
- EventProcessorCaltech101RPG
- EventProcessorDAVIS
- EventProcessorProphesee
- EventProcessorRAMNET

For more detailed usage, refer to the documentation of each class and method.
"""

from collections import defaultdict
import hdf5plugin
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from abc import ABC, abstractmethod

class EventProcessor(ABC):
    """
    Abstract base class for event processors.
    
    This class defines the interface for all event processors and implements
    common functionality.
    """

    def __init__(self, **kwargs):
        """
        Initialize the EventProcessor.

        Args:
            **kwargs: Arbitrary keyword arguments.
                bin_size (int): Size of a bin (temporal window) in milliseconds. Default is 33.
        """
        self.events_raw = None
        self.events_dict = None
        self.events_buff = {}
        self.bin_size = kwargs.get('bin_size', 33)
        self.evs_dataset_type = 'generic'
        self.size_y = None
        self.size_x = None
        self.init_tstamp = None
        self.current_t = 0.0

    @abstractmethod
    def __events_to_generic__(self, evs):
        """
        Convert events to a generic format.

        Args:
            evs: Events in the original format.

        Returns:
            dict: Events in the generic format {x, y, t, p}.
        """
        pass

    @abstractmethod
    def read_next_event_packets(self, num_events) -> dict:
        """
        Read the next batch of event packets.

        Args:
            num_events (int): Number of event packets to read.

        Returns:
            dict: Events in the generic format, or -1 if no more events.
        """
        pass

    def reset(self):
        """
        Reset the event processor to the initial state.
        """
        self.current_t = 0.0
        self.events_buff = {}
    
    def read_all_events(self) -> dict:
        """
        Read all events from the source.

        Returns:
            dict: All events in the generic format.
        """
        ev_dict_res = {k: np.array([], dtype=int) for k in ['x', 'y', 't', 'p']}
        while True:
            ev_dict_ = self.read_next_event_packets(1)
            if ev_dict_ == -1:
                break
            ev_dict_res = concatenate_event_dicts(ev_dict_res, ev_dict_)
        return ev_dict_res

    def get_next_bins(self, num_bins=1, **kwargs) -> dict:
        """
        Read the next temporal window of events.

        Args:
            num_bins (int): Number of bins to read.
            **kwargs: Additional arguments passed to read_events_until.

        Returns:
            dict: Events in the generic format.
        """
        t_stop_ = self.current_t + (self.bin_size * num_bins)
        return self.read_events_until(t_stop_, **kwargs)

    def read_events_until(self, t_stop, **kwargs):
        """
        Read events up to a specified time.

        Args:
            t_stop (float): Stop time in milliseconds.
            **kwargs: Additional arguments.

        Returns:
            dict: Events in the generic format, or -1 if no more events.
        """
        t_stop_us = t_stop * 1e6 if self.evs_dataset_type == "caltech101rpg" else t_stop
        ev_dict_res = {k: np.array([], dtype=int) for k in ['x', 'y', 't', 'p']}
        flag_buff_checked = False
        flag_no_read_more = kwargs.get("flag_no_read_more", False)

        while True:
            if ((not self.events_buff) or flag_buff_checked) and not flag_no_read_more:
                ev_dict_ = self.read_next_event_packets(1)
            else:
                ev_dict_ = self.events_buff
                flag_buff_checked = True

            if ev_dict_ == -1 or not ev_dict_:
                return -1

            events_ns_ = ev_dict_.get('t')
            last_event_us = events_ns_[-1] if len(events_ns_) > 0 else 0

            if last_event_us >= t_stop_us:
                t_mask = ev_dict_['t'] < t_stop_us
                ev_dict_last = {k: np.asarray(v[t_mask]) for k, v in ev_dict_.items()}
                ev_dict_res = concatenate_event_dicts(ev_dict_res, ev_dict_last)
                self.events_buff = {k: np.asarray(v[~t_mask]) for k, v in ev_dict_.items()}
                break
            else:
                ev_dict_res = concatenate_event_dicts(ev_dict_res, ev_dict_)
                if flag_no_read_more:
                    self.current_t = ev_dict_res['t'][-1]
                    self.events_buff = {}
                    return ev_dict_res

        self.current_t = self.events_buff['t'][0] * (1e-6 if self.evs_dataset_type == "caltech101rpg" else 1)
        return ev_dict_res

    def filter_events_to_img(self, x, y, t, p):
        """
        Filter events to fit within image bounds.

        Args:
            x (np.ndarray): X coordinates of events.
            y (np.ndarray): Y coordinates of events.
            t (np.ndarray): Timestamps of events.
            p (np.ndarray): Polarities of events.

        Returns:
            tuple: Filtered x, y, t, p arrays.
        """
        mask = (x < self.size_x) & (y < self.size_y) & (x >= 0) & (y >= 0)
        return x[mask], y[mask], t[mask], p[mask]


def filter_events_by_polarity(ev_dict, p_type):
    """
    Filter events by polarity.

    Args:
        ev_dict (dict): Events dictionary.
        p_type (int): Polarity type to filter (0 or 1).

    Returns:
        dict: Filtered events dictionary.
    """
    mask = ev_dict['p'] == p_type
    return {k: np.asarray(v[mask]) for k, v in ev_dict.items()}


def concatenate_event_dicts(ev_dict_a, ev_dict_b):
    """
    Concatenate two event dictionaries.

    Args:
        ev_dict_a (dict): First event dictionary.
        ev_dict_b (dict): Second event dictionary.

    Returns:
        dict: Concatenated event dictionary.
    """
    return {k: np.concatenate((ev_dict_a[k], ev_dict_b[k])) for k in ev_dict_a}


class EventProcessorVid2e(EventProcessor):
    """
    Event processor for Vid2e dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evs_path = kwargs.get('evs_path')
        self.imgs_path = kwargs.get('imgs_path')
        self.tstmp_path = kwargs.get('tstmp_path')
        self.evs_dataset_type = 'vid2e'
        self.ev_files_list = None
        self.img_files_list = None
        self.ev_files_iter = None
        self.img_files_iter = None
        self.current_ev_file_idx = -1
        self.current_frame_idx = 0
        self.__init_vid2e_events_file__()

    def __init_vid2e_events_file__(self):
        self.__read_ev_files__()
        self.__read_img_files__()
        self.__read_img_timestamp__()
        img_0 = Image.open(self.img_files_list[0])
        self.size_y, self.size_x = img_0.height, img_0.width
        self.init_tstamp = self.__get_init_ev_tstamp__()
        self.current_t += self.init_tstamp

    def __read_ev_files__(self):
        self.ev_files_list = sorted([ev_f for ev_f in Path(self.evs_path).iterdir() if ev_f.is_file()])
        self.ev_files_iter = iter(self.ev_files_list)

    def __read_img_files__(self):
        self.img_files_list = sorted([img_f for img_f in Path(self.imgs_path).iterdir() if img_f.is_file() and img_f.suffix in [".jpg", ".png"]])
        self.img_files_iter = iter(self.img_files_list)

    def __read_img_timestamp__(self):
        self.tstmp_path = Path(self.tstmp_path) if self.tstmp_path else Path(self.imgs_path)
        tstamp_file = self.tstmp_path / 'timestamp.txt'
        if not tstamp_file.is_file():
            tstamp_file = self.tstmp_path / 'timestamps.txt'
        self.tstamps_list = pd.read_csv(tstamp_file, header=None)[0].tolist()

    def __get_init_ev_tstamp__(self):
        for i, ev_file in enumerate(self.ev_files_list):
            evs_ = [self.read_ev_file(ev_file)]
            evs_np = {k: v for d in evs_ for k, v in d.items()}
            ev_dict_ = self.__events_to_generic__(evs_np)
            tstamps = ev_dict_.get('t')
            if len(tstamps) > 0:
                return tstamps[0]
        raise ValueError("No events found in any file")

    def __events_to_generic__(self, evs_np):
        timestamps, x, y, polarities = evs_np.get('t'), evs_np.get('x'), evs_np.get('y'), evs_np.get('p')
        polarities = ((polarities + 1) / 2).astype(int)
        return {'x': x, 'y': y, 't': timestamps, 'p': polarities}

    def reset(self):
        super().reset()
        self.ev_files_iter = iter(self.ev_files_list)
        self.current_ev_file_idx = -1
        self.current_frame_idx = 0

    def read_next_event_packets(self, num_packets, skip=False):
        if skip:
            _ = [ff for _, ff in zip(range(num_packets), self.ev_files_iter)]
            self.current_ev_file_idx += len(_)
            return []
        evs = [self.read_ev_file(ff) for _, ff in zip(range(num_packets), self.ev_files_iter)]
        self.current_ev_file_idx += len(evs)
        if not evs:
            return -1

        evs_np = defaultdict(list)
        for d in evs:
            for k, v in d.items():
                evs_np[k].extend(v)

        # List to array
        for k, v in evs_np.items():
            evs_np[k] = np.array(v)

        return self.__events_to_generic__(evs_np)

    @staticmethod
    def read_ev_file(ev_file):
        return np.load(str(ev_file))

    @staticmethod
    def read_frame_file(frame_file):
        return Image.open(str(frame_file)).convert("RGB")


class EventProcessorDVSVoltmeter(EventProcessor):
    """
    Event processor for DVS Voltmeter dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evs_path = kwargs.get('evs_path')
        self.imgs_path = kwargs.get('imgs_path')
        self.evs_dataset_type = 'dvsvoltmeter'
        self.init_tstamp = self.__get_init_ev_tstamp__()
        self.current_t += self.init_tstamp

    def __get_init_ev_tstamp__(self):
        evs_dict = self.read_next_event_packets(1)
        return evs_dict['t'][0]

    def __events_to_generic__(self, evs_np):
        timestamps, x, y, polarities = evs_np[:, 0].astype(np.float32), evs_np[:, 1].astype(np.int32), evs_np[:, 2].astype(np.int32), evs_np[:, 3].astype(int)
        return {'x': x, 'y': y, 't': timestamps, 'p': polarities}

    def read_next_event_packets(self, num_packets):
        evs_np = np.loadtxt(str(self.evs_path)).astype(np.float32)
        return self.__events_to_generic__(evs_np)


class EventProcessorTimelens(EventProcessorVid2e):
    """
    Event processor for Timelens dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evs_dataset_type = 'timelens'

    def __events_to_generic__(self, evs_np):
        timestamps, x, y, polarities = evs_np.get('t'), evs_np.get('x'), evs_np.get('y'), evs_np.get('p')
        return {'x': x, 'y': y, 't': timestamps, 'p': polarities}


class EventProcessorRVT(EventProcessor):
    """
    Event processor for RVT dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evs_path = kwargs.get('evs_path')
        self.evs_dataset_type = 'rvt'
        self.h5_reader = None
        self.h5_len = None
        self.divider = None
        self.chunks = None
        self.bin_size = kwargs.get('bin_size', 10000)
        self.bin_idx = 0
        self.__init_rvt_events_file__()

    def __init_rvt_events_file__(self):
        self.h5_reader = self.H5Reader(self.evs_path)
        self.events_raw = self.h5_reader.h5f['events']
        self.h5_len = len(self.events_raw['t'])
        self.divider = self.events_raw['divider']
        self.size_y = self.h5_reader.height
        self.size_x = self.h5_reader.width
        self.init_tstamp = self.events_raw['t'][0]
        self.current_t += self.init_tstamp

    def reset(self):
        super().reset()
        self.bin_idx = 0

    def read_next_event_packets(self, num_packets=1):
        idx_start = self.bin_idx
        idx_end = self.bin_idx + (self.bin_size * num_packets)
        
        x_array = self.events_raw['x'][idx_start:idx_end]
        y_array = self.events_raw['y'][idx_start:idx_end]
        t_array = self.events_raw['t'][idx_start:idx_end]
        p_array = self.events_raw['p'][idx_start:idx_end]

        if len(t_array) == 0:
            return -1

        self.bin_idx = idx_end

        ev_data = {
            'x': x_array,
            'y': y_array,
            't': t_array,
            'p': p_array
        }

        # Remove duplicate events
        df = pd.DataFrame(ev_data).drop_duplicates()
        ev_data = dict(zip(df.columns, df.values.T))

        return self.__events_to_generic__(ev_data)

    def __events_to_generic__(self, evs_np):
        timestamps = evs_np['t']
        x = evs_np['x']
        y = evs_np['y']
        polarities = np.clip(evs_np['p'], a_min=0, a_max=None)  # Ensure polarities are 0 or positive
        return {'x': x, 'y': y, 't': timestamps, 'p': polarities}

    class H5Reader:
        def __init__(self, h5_file: Path):
            self.h5f = h5py.File(str(h5_file), 'r')
            self.height = self.h5f['events']['height'][()].item()
            self.width = self.h5f['events']['width'][()].item()
            self.all_times = None

        @property
        def time(self) -> np.ndarray:
            if self.all_times is None:
                self.all_times = np.asarray(self.h5f['events']['t'])
                self._correct_time(self.all_times)
            return self.all_times

        @staticmethod
        def _correct_time(time_array: np.ndarray):
            assert time_array[0] >= 0
            time_last = 0
            for idx, time in enumerate(time_array):
                if time < time_last:
                    time_array[idx] = time_last
                else:
                    time_last = time


class EventProcessorCaltech101(EventProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events_file = kwargs.get('evs_file')
        self.evs_dataset_type = 'caltech101'
        self.time_window_ms = kwargs.get('time_window_ms', 33)  # ~30 fps
        self.__init_vars__()

    def __init_vars__(self):
        self.__read_ev_files__()
        self.size_y = self.events_buff['y'].max()
        self.size_x = self.events_buff['x'].max()
        self.init_tstamp = 0
        self.current_t = 0

    def __read_ev_files__(self):
        raw_evs = self.read_ev_file(self.events_file)
        self.__events_to_buffer__(raw_evs)

    def __events_to_buffer__(self, raw_evs):
        # Implement conversion of raw events to buffer format
        pass

    def __events_to_generic__(self, evs_np):
        return evs_np

    def reset(self):
        super().reset()
        self.__read_ev_files__()

    def read_next_event_packets(self, num_packets, skip=False):
        evs_np = self.get_next_bins(num_bins=num_packets, flag_no_read_more=True)
        if evs_np == -1:
            return -1
        evs_np['t'] = evs_np['t'] * 1e-9  # Convert nanoseconds to seconds
        return self.__events_to_generic__(evs_np)

    @staticmethod
    def read_ev_file(ev_file):
        with open(ev_file, 'rb') as file_handle:
            raw_data = np.fromfile(file_handle, dtype=np.uint8)
        return raw_data


class EventProcessorCaltech101RPG(EventProcessorCaltech101):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evs_dataset_type = 'caltech101rpg'

    def __events_to_buffer__(self, raw_evs):
        x_ = raw_evs[:, 0].astype(np.uint16)
        y_ = raw_evs[:, 1].astype(np.uint16)
        t_ = raw_evs[:, 2].astype(np.float64) * 1e9  # Convert seconds to nanoseconds
        p_ = raw_evs[:, 3].astype(int)

        evs_np = {'x': x_, 'y': y_, 't': t_, 'p': p_}
        self.events_buff = self.__events_to_generic__(evs_np)

    def __events_to_generic__(self, evs_np):
        timestamps, x, y, polarities = evs_np['t'], evs_np['x'], evs_np['y'], evs_np['p']
        polarities = ((polarities + 1) / 2).astype(int)  # Convert (-1, 1) to (0, 1)
        return {'x': x, 'y': y, 't': timestamps, 'p': polarities}

    def reset(self):
        super().reset()
        self.bin_idx = 0

    @staticmethod
    def read_ev_file(ev_file):
        return np.load(str(ev_file)).astype(np.float32)


class EventProcessorDAVIS(EventProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events_file = kwargs.get('evs_file')
        self.evs_dataset_type = 'davis'
        self.aedat_file_object = None
        self.__init_davis_events_file__()

    def __init_davis_events_file__(self):
        from dv import AedatFile
        self.aedat_file_object = AedatFile(self.events_file)
        self.size_y = self.aedat_file_object['events'].size_y
        self.size_x = self.aedat_file_object['events'].size_x
        self.init_tstamp = int(self.aedat_file_object['events']._stream_info['tsOffset'])

    def __events_to_generic__(self, evs_np):
        return {
            'x': evs_np['x'],
            'y': evs_np['y'],
            't': evs_np['timestamp'],
            'p': evs_np['polarity']
        }

    def read_next_event_packets(self, num_packets):
        evs = [packet for _, packet in zip(range(num_packets), self.aedat_file_object['events'].numpy())]
        evs_np = np.hstack(evs)
        return self.__events_to_generic__(evs_np)


class EventProcessorProphesee(EventProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events_file = kwargs.get('evs_file')
        self.evs_dataset_type = 'prophesee'
        self.wizard = None
        self.file_encoding = kwargs.get('file_encoding', 'evt3')
        self.chunk_size = kwargs.get('chunk_size', 4096)
        self.time_window_ms = kwargs.get('time_window_ms', 33)  # ~30 fps
        self.__init_prophesee_events_file__()

    def __init_prophesee_events_file__(self):
        from expelliarmus import Wizard
        self.wizard = Wizard(encoding=self.file_encoding, fpath=self.events_file)
        self.wizard.set_chunk_size(chunk_size=self.chunk_size)
        self.wizard.set_time_window(time_window=self.time_window_ms)

    def __events_to_generic__(self, evs_np):
        return {
            'x': evs_np['x'],
            'y': evs_np['y'],
            't': evs_np['t'],
            'p': evs_np['p']
        }

    def read_next_event_packets(self, num_packets):
        evs = [next(self.wizard.read_chunk()) for _ in range(num_packets)]
        evs_np = np.hstack(evs)
        return self.__events_to_generic__(evs_np)

    def read_next_time_window(self, num_windows=1):
        evs = [next(self.wizard.read_time_window()) for _ in range(num_windows)]
        evs_np = np.hstack(evs)
        return self.__events_to_generic__(evs_np)


class EventProcessorRAMNET(EventProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.folder_dir = kwargs.get('evs_path')
        self.evs_dataset_type = 'ramnet'
        self.events_files = []
        self.ev_files_list = []
        self.current_ev_file_idx = 0
        self.__init_RAMNET_events_file__()

    def __init_RAMNET_events_file__(self):
        self.size_x = 512
        self.size_y = 256

        self.events_files.append(str(self.folder_dir))
        self.ev_files_list.append(self.folder_dir.name)

        self.ev_files_list = sorted(self.events_files)
        self.events_files_iter = iter(self.events_files)

        self.init_tstamp = self.__get_init_ev_tstamp__()
        self.current_t += self.init_tstamp

    def __get_init_ev_tstamp__(self):
        for i in range(len(self.ev_files_list)):
            evs_ = [self.read_ev_file(self.ev_files_list[i])]
            evs_np = {k: v for d in evs_ for k, v in d.items()}
            ev_dict_ = self.__events_to_generic__(evs_np)
            tstamps = ev_dict_.get('t')
            if len(tstamps) > 0:
                return tstamps[0]
        return 0

    def __events_to_generic__(self, evs):
        return {
            'x': evs['x'],
            'y': evs['y'],
            't': evs['t'],
            'p': evs['p']
        }

    def read_next_event_packets(self, num_packets=1) -> dict:
        evs = [self.read_ev_file(ff) for _, ff in zip(range(num_packets), self.events_files_iter)]
        if len(evs) == 0:
            return -1
        self.current_ev_file_idx += len(evs)

        evs_np = defaultdict(list)
        for d in evs:
            for k, v in d.items():
                evs_np[k].extend(v)

        # List to array
        for k, v in evs_np.items():
            evs_np[k] = np.array(v)

        return self.__events_to_generic__(evs_np)

    @staticmethod
    def read_ev_file(ev_file):
        return np.load(str(ev_file))


class EventProcessorShapes(EventProcessor):
    '''
    Event processor for shapes dataset (https://www.ifi.uzh.ch/en/rpg/software_datasets/davis_datasets.html) for shapes translation motion.
    Files: events.txt, images.txt, groundtruth.txt (this one is optional)
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events_file = kwargs.get('evs_file')
        self.evs_dataset_type = 'shapes'

    def __events_to_generic__(self, evs_np):
        return {
            'x': evs_np['x'],
            'y': evs_np['y'],
            't': evs_np['t'],
            'p': evs_np['p']
        }
    
    def read_next_event_packets(self, num_packets=1) -> dict:
        evs = [self.read_ev_file(ff) for _, ff in zip(range(num_packets), self.events_files_iter)]
        if len(evs) == 0:
            return -1
        self.current_ev_file_idx += len(evs)

        evs_np = defaultdict(list)
        for d in evs:
            for k, v in d.items():
                evs_np[k].extend(v)

        # List to array
        for k, v in evs_np.items():
            evs_np[k] = np.array(v)

        return self.__events_to_generic__(evs_np)


class EventProcessorESFP(EventProcessor):
    """
    Event processor for ESFP dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evs_path = kwargs.get('evs_path')
        self.evs_dataset_type = 'esfp'
        self.h5_len = None
        self.chunks = None
        self.init_tstamp = 0
        self.current_t = 0
        self.bin_size = kwargs.get('bin_size', 10000)
        self.bin_idx = 0
        self.__init_esfp_events_file__()

    def __init_esfp_events_file__(self):
        self.events_raw = self.read_ev_file(self.evs_path)
        self.size_y = 720  # Dataset resolution
        self.size_x = 1280
        self.h5_len = len(self.events_raw['t'])
        #print(f"{len(self.events_raw['t'])}, {len(self.events_raw['x'])}, {len(self.events_raw['y'])}, {len(self.events_raw['p'])}")
        self.init_tstamp = self.events_raw['t'][0]
        self.current_t += self.init_tstamp

    def __events_to_generic__(self, evs_np):
        timestamps = evs_np['t']
        x = evs_np['x']
        y = evs_np['y']
        polarities = np.clip(evs_np['p'], a_min=0, a_max=None)  # Ensure polarities are 0 or positive
        return {'x': x, 'y': y, 't': timestamps, 'p': polarities}

    def reset(self):
        super().reset()
        self.bin_idx = 0

    def read_next_event_packets(self, num_packets=1):
        idx_start = self.bin_idx
        idx_end = self.bin_idx + (self.bin_size * num_packets)
        
        x_array = self.events_raw['x'][idx_start:idx_end]
        y_array = self.events_raw['y'][idx_start:idx_end]
        t_array = self.events_raw['t'][idx_start:idx_end]
        p_array = self.events_raw['p'][idx_start:idx_end]

        if len(t_array) == 0:
            return -1

        self.bin_idx = idx_end

        ev_data = {
            'x': x_array,
            'y': y_array,
            't': t_array,
            'p': p_array
        }

        # Remove duplicate events
        df = pd.DataFrame(ev_data).drop_duplicates()
        ev_data = dict(zip(df.columns, df.values.T))

        return self.__events_to_generic__(ev_data)

    def read_ev_file(self, ev_file):
        with h5py.File(ev_file, 'r') as f:
            # Extract properties

            chunks_dict = {}
            source_dsets = [f['x'], f['y'], f['t'], f['p']]
            for key in f.keys():
                source_dset = f[key]
                dtype = source_dset.dtype
                shape = source_dset.shape
                maxshape = source_dset.maxshape
                compression = source_dset.compression
                compression_opts = source_dset.compression_opts
                shuffle = source_dset.shuffle
                chunks = source_dset.chunks
                chunks_dict[key] = chunks

                #print(f"dtype: {dtype}, shape: {shape}, maxshape: {maxshape}, compression: {compression}, compression_opts: {compression_opts}, shuffle: {shuffle} chunks: {chunks}")

            self.chunks = chunks_dict

            return {
                'x': f['x'][:],
                'y': f['y'][:],
                't': f['t'][:],
                'p': f['p'][:]
            }


if __name__ == '__main__':
    # Event-Camera-Dataset
    #event_processor_caltech101 = EventProcessorCaltech101RPG(evs_file="/home/user_fiftyone/datasets/N-Caltech101/rpg_event_representation_learning/original/training/Faces_easy/Faces_easy_0.npy")
    # ESFP
    event_processor_esfp = EventProcessorESFP(evs_file="/home/user/datasets/ESFP-Real/realworld_dataset_clean/test/bottle_26_10/events.h5")

    print(event_processor_esfp.get_next_bins(1))
