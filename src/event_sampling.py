# Copyright: Andreu Girbau-Xalabarder, 2025
# We provide the code as-is,
# without warranty of any kind, express or implied.

# Main file that contains the class EventSampler and its children
import torch

from collections import defaultdict
import math
import multiprocessing
import numpy as np
import os
from pathlib import Path
from PIL import Image
import random
from scipy.ndimage import zoom

from config import get_parser
from prob_models import ProbIdentity, AccumGridProb, PoissonProb

from utils import EventFileSaver, filter_events_to_img, get_percentile_rank, tranform_pdf_to_percentile_rank_nearest, count_decimal_positions, load_prior
import utils as ut

# To avoid "ModuleNotFoundError: No module named 'external'" error
# append "external" folder to the sys path, which is in the parent directory
import sys
external_path = Path(__file__).resolve().parent.parent / "external"
sys.path.append(str(external_path.parent))

from external.processor import EventProcessorTimelens, EventProcessorCaltech101RPG, EventProcessorRVT, EventProcessorESFP
from external.downsampling_methods import event_based_downsampling as evDownNavi

import time


# Common variables
# It's outside "__main__" due to multiprocessing not sharing memory of the elements inside __main__ for the "spawn" start method
# https://newbedev.com/workaround-for-using-name-main-in-python-multiprocessing

parser = get_parser()
sysargs = parser.parse_args()

# Experiment configuration
exp_cfg = ut.load_yaml(sysargs.cfg)
# Read variables from the experiment config file
dataset_name = sysargs.dataset_name or exp_cfg.dataset.name
dataset_cfg_path = (Path('/home/user/app/config/datasets') / dataset_name).with_suffix('.yaml')
dataset_cfg = ut.load_yaml(dataset_cfg_path)
# Merge dicts
exp_cfg = ut.merge_dicts(exp_cfg, dataset_cfg)
exp_cfg.dataset.split = sysargs.split or exp_cfg.dataset.split  # As a hotfix, but should be implemented properly
exp_cfg.flags.debug = exp_cfg.flags.debug or sysargs.d  # True if any of them is True
paths_dict = ut.get_cfg_sampling_paths(exp_cfg) if not exp_cfg.flags.debug else ut.get_cfg_paths(exp_cfg)
# Merge with CLI args
config = ut.merge_configs(exp_cfg, sysargs)
# Some final (hopefully temporary) adjustments
# Dataset type
config['dataset_name'] = config['name']
config['dataset_type'] = config['type']
# Paths
config['seqs_path'] = paths_dict['seqs_path']
#config['ev_path'] = paths_dict['evs_path']
config['ev_save_path'] = paths_dict['evs_save_path']
config['img_path'] = paths_dict['img_path']
config['tstmp_path'] = paths_dict['tstmp_path']
config['prior_path'] = paths_dict['prior_path']
# Variables
config['sampler_type'] = config['sampler']
config['prob_init'] = float(f"0.{config['prob_init']}") if float(config['prob_init']) < 10 else 1

if config['debug'] and config['dataset_type'] == 'caltech101':
    config['ev_path'] = paths_dict['evs_path']

# Set random seeds for reproducibility
np.random.seed(int(config['num_exp']))
random.seed(int(config['num_exp']))

# Load prior distribution (for RVT dataset)
config['prior'] = None
config['prior_pos'] = None

# Adjust configuration for Gen1 dataset
if exp_cfg.flags.prior:
    if config['dataset_name'] == 'NCaltech101':
        config['prior'] = ut.generate_gaussian_prior(10, 10, 0.8, 0.8)
    elif config['dataset_name'] == 'gen1':
        config['prior'] = ut.load_prior(config["prior_path"])
    elif config['dataset_name'] == 'esfp':
        config['prior'] = np.ones((10, 10))
    else:
        config['alpha_prior'] = 0
else:
    config['alpha_prior'] = 0


class EventSampler:
    """Base class for event sampling algorithms."""

    def __init__(self, **kwargs):
        # Paths
        self.evs_write_path = kwargs.get("evs_write_path", None)  # Root path for saving sampled events
        
        # Probability grids
        self.G = None  # Grid of acceptance probabilities
        self.T = None  # Grid with timestamps of previous events per position
        self.t_diff = None  # Grid with time differences between events
        self.grid_shape = None  # Idk, it's grid_size + 1
        
        # Probability model
        self.prob_model = kwargs.get("prob_model", ProbIdentity())  # Default to identity model if not specified
        self.choices = [True, False]
        
        # Sampling parameters
        self.sep_pols = kwargs.get("sep_pols", False)  # Whether to separate polarities
        self.grid_size = kwargs.get("sensor_size", (5000, 5000))  # Sensor size (y, x)
        self.sampling_rate = kwargs.get("prob_init", 0.5)  # Initial probability of event acceptance
        self.prob_max = kwargs.get("prob_max", 1)  # Maximum probability in the grid
        self.flag_decay = kwargs.get('decay', False)
        
        # Event counters
        self.num_evs_orig = [1, 1]  # Count of original events [OFF, ON]
        self.num_evs_smpl = [1, 1]  # Count of sampled events [OFF, ON]
        
        # Sampling control
        self.rel_original_samples = None  # Ratio of sampled to original events
        self.percentiles = [0, 10, 25, 50, 75, 90, 100]
        self.pdf_percentiles = [self.sampling_rate]
        
        # Lambda parameters for event acceptance/rejection
        self.l_discard = config['l_reject']
        self.l_accept = config['l_accept']
        
        # Other parameters
        self.k_size = 25
        self.max_events = None  # For future use (e.g., constant event rate)

        # Variables
        self.eps = 1e-8  # Small value to avoid division by zero

        # Debug variables
        self.debug = kwargs.get('debug', False)
        self.pdf_idx = [0, 0]
        self.num_events_processed_ = 0
        self.time_spent_ = time.time()
        self.time_pdf_est_ = 0
        self.time_eval_ = 0
        
        # Initialize probability grid
        self._init_grid_probs_()

    def __call__(self, *args, **kwargs):
        """To be implemented in child classes."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def _init_grid_probs_(self):
        """Initialize the grid of probabilities G(y, x) = p(t)."""
        if self.grid_size is not None:
            # G[0] -> OFF events, G[1] -> ON events
            self.G = np.ones((2, self.grid_size[0]+1, self.grid_size[1]+1), dtype=int) * self.sampling_rate
            self.T = np.zeros_like(self.G, dtype=np.float64)
            self.t_diff = np.ones_like(self.G, dtype=np.float64) * 1e6  # Initialize with a large value
            self.grid_shape = self.G.shape

    def update_prior(self, new_prior):
        # Dummy method to be implemented in child classes
        pass

    def filter_event(self, prob_ev):
        """Decide whether to accept or reject an event based on its probability."""
        #t_start = time.perf_counter()
        #probabilities = [prob_ev, 1-prob_ev]
        #res = random.choices(self.choices, probabilities)[0]
        res = random.random() < prob_ev
        #self.time_eval_ += (time.perf_counter() - t_start)
        return res

    def eval_event(self, pos_int, t, p=0):
        """Evaluate whether to accept or reject an event. To be implemented in child classes."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def update_grid_time(self, pos, new_t, p=0):
        """Update the time grid with the current event."""
        self.T[p][pos] = new_t

    def update_grid_time_diff(self, pos, new_t, p=0):
        """Update the time grid with the current event."""
        self.t_diff[p][pos] = new_t - self.T[p][pos]

    def update_grid_probs(self, pos, new_p, p=0):
        """Update the probability grid with a new probability."""
        self.G[p][pos] = new_p

    def sample_events(self, events):
        """Sample events based on the current sampling strategy."""
        new_events_dict = {"x": [], "y": [], "t": [], "p": []}
        x, y, t, p = events["x"], events["y"], events["t"], events["p"]
        x, y, t, p = filter_events_to_img(self.grid_size, x, y, t, p)

        for x_, y_, t_, p_ in zip(x, y, t, p):
            x_int_, y_int_, p_int_ = int(x_), int(y_), int(p_)
            
            if not self.sep_pols:
                p_int_ = 0  # Don't separate polarities

            t_start = time.perf_counter()
            is_accept = self.eval_event((y_int_, x_int_), t_, p_int_)
            self.time_pdf_est_ += (time.perf_counter() - t_start)

            self.num_evs_orig[p_int_] += 1

            # Cap accepted events based on sampling rate
            # Here is the thing, if one polarity dominates, (e.g. 90-10), the other polarity will dissapear if we only take into account the total amount of events,
            # which, in contrast, is the most realistic thing (?)
            #self.rel_original_samples = self.num_evs_smpl[p_int_] / self.num_evs_orig[p_int_] # Ratio per polarity
            self.rel_original_samples = np.sum(self.num_evs_smpl) / np.sum(self.num_evs_orig)  # Total ratio
            if self.rel_original_samples > self.sampling_rate:
                is_accept = False

            if is_accept:
                self.num_evs_smpl[p_int_] += 1
                new_events_dict["x"].append(x_)
                new_events_dict["y"].append(y_)
                new_events_dict["t"].append(t_)
                new_events_dict["p"].append(p_)
                if self.flag_decay:
                    self.update_grid_time_diff((y_int_, x_int_), t_, p_int_)  # Update time diff grid for all events
                    self.update_grid_time((y_int_, x_int_), t_, p_int_)  # Update time grid for accepted events
            else:
                if self.flag_decay:
                    self.update_grid_time_diff((y_int_, x_int_), t_, p_int_)  # Update time diff grid for all events

        # Convert lists to numpy arrays, preserving original data types
        for key in new_events_dict:
            new_events_dict[key] = np.array(new_events_dict[key], dtype=events[key].dtype)

        return new_events_dict


class EventSamplerProbabilistic(EventSampler):
    """Probabilistic event sampler that uses a score matrix to determine event acceptance."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_matrix = None
        self.percentile_matrix = None
        self.alpha_prior = kwargs.get('alpha_prior', 0)
        self._init_prior()
        self._init_score_matrix()

    def _init_score_matrix(self):
        """Initialize the score matrix with the initial sampling rate."""
        self.score_matrix = np.full(self.grid_shape, self.sampling_rate, dtype=float)
        self.percentile_matrix = np.full(self.grid_shape, self.sampling_rate, dtype=float)
        
        # Initialize score matrix with a uniform distribution
        for p in [0, 1]:
            self.pdf_to_score(self.score_matrix[p], p)

    def _init_prior(self):
        """Initialize and process the prior distribution if applicable."""
        self.prior_pdf = np.ones(self.grid_shape[1:], dtype=float)
        self.prior_score = np.ones(self.grid_shape[1:], dtype=float)
        
        if self.alpha_prior > 0:
            self.prior_pdf = config['prior']
            self._scale_prior(self.prior_pdf)
            self.prior_score = self._process_prior(self.prior_pdf)
            self._pad_prior()

    def _scale_prior(self, prior):
        """Scale prior size to match the score matrix."""
        self.prior_pdf = zoom(prior, (self.grid_shape[1] / prior.shape[0], self.grid_shape[2] / prior.shape[1]), order=0)

    def _process_prior(self, prior):
        """Process the prior distribution to create a score."""
        prior_score = (prior / prior.max()) + self.sampling_rate
        return (prior_score / np.mean(prior_score)) * self.sampling_rate

    def _pad_prior(self):
        """Pad the prior to match the score matrix dimensions."""
        pad_size = np.array(self.grid_shape[1:]) - np.array(self.prior_pdf.shape)
        padding = ((0, pad_size[0]), (0, pad_size[1]))
        self.prior_pdf = np.pad(self.prior_pdf, padding, 'edge')
        self.prior_score = np.pad(self.prior_score, padding, 'edge')

    def update_prior(self, new_prior):
        """Update the prior distribution."""
        self.prior_pdf = new_prior
        self._scale_prior(self.prior_pdf)
        self.prior_score = self.scale_pdf(self.prior_pdf)
        self._pad_prior()
        self._init_score_matrix()

    def scale_pdf_old(self, pdf):
        """Scale the PDF to create a score."""
        score = (pdf / pdf.max()) + self.sampling_rate
        return (score / np.mean(score)) * self.sampling_rate

    def scale_pdf(self, pdf):
        """Scale the PDF to create a score."""
        pdf_max_min = (pdf - np.min(pdf)) / (np.max(pdf) - np.min(pdf) + self.eps)
        pdf_displaced_mean =  pdf_max_min + (self.sampling_rate - np.mean(pdf_max_min))  # Technically mean(pdf) is 1/N, where N=Area of the sensor.
        return pdf_displaced_mean

    def save_pdf(self, pdf, save_dir, p=0):
        # Save weight matrix in the form of an image for visualization
        norm_mat = pdf / pdf.max()
        np_img = (norm_mat * 255).astype(np.uint8)
        image = Image.fromarray(np_img)

        save_name = f'{save_dir}/{p}_{self.pdf_idx[p]}.jpg'

        os.makedirs(save_dir, exist_ok=True)
        image.save(save_name)
        self.pdf_idx[p] += 1
    
    def pdf_to_score_old(self, p):
        """Convert PDF to score, incorporating prior if applicable."""
        ev_pdf = self.prob_model.E_weights[p]
        self.rel_original_samples = self.num_evs_smpl[p] / self.num_evs_orig[p]
        
        pdf_with_prior = (self.alpha_prior * self.prior_pdf) + ((1 - self.alpha_prior) * ev_pdf)
        self.score_matrix[p] = self.scale_pdf(pdf_with_prior)

    def pdf_to_score(self, ev_pdf, p):
        """Convert PDF to score, incorporating prior if applicable.
        Characteristics: 
        - Displace the mean to the sampling rate (alpha)
        - Return a probability of acceptance based on a Bernoulli distribution (p, 1-p)
        """        
        pdf_with_prior = self.prior_pdf * ev_pdf
        pdf_scaled = self.scale_pdf(pdf_with_prior)

        p1_ = 5
        p2_ = 0.5
        score = 1 / (1 + np.exp(-p1_ * (pdf_scaled - p2_)))
        
        self.score_matrix[p] = score

    def pdf_to_percentile(self, ev_pdf, p):
        """Convert PDF to percentile."""        
        pdf_with_prior = (self.alpha_prior * self.prior_pdf) + ((1 - self.alpha_prior) * ev_pdf)
        
        # Flatten the PDF and sort it
        flat_pdf = pdf_with_prior.ravel()
        sorted_indices = np.argsort(flat_pdf)
        
        # Calculate cumulative sum of sorted PDF
        cumsum = np.cumsum(flat_pdf[sorted_indices])
        
        # Calculate percentiles
        percentiles = np.zeros_like(flat_pdf)
        percentiles[sorted_indices] = cumsum / cumsum[-1]
        
        # Reshape the percentiles back to the original shape
        percentiles = percentiles.reshape(pdf_with_prior.shape)
        
        # Assign percentiles to percentile matrix
        self.percentile_matrix[p] = self.scale_pdf(percentiles)

    def eval_event(self, pos_int, t, p=0):
        """Evaluate whether to accept or reject an event."""
        # Initialize time if necessary
        if self.prob_model.t_init[p] == 0:
            self.prob_model.set_init_time(t, p=p)

        # Update probability model
        self.prob_model(pos_int=pos_int, t=t, p=p)

        # Update score matrix if weights have changed
        if self.prob_model.flag_weights_update:
            ev_pdf = self.prob_model.E_weights[p]
            self.pdf_to_score(ev_pdf, p)
            #self.pdf_to_percentile(p)
            if self.debug:
                if self.prob_model.dataset_type == 'caltech101':
                    save_dir = f'/home/user/app/tmp/poisson/flamingo/1'
                elif self.prob_model.dataset_type == 'cars':
                    save_dir = f'/home/user/app/tmp/poisson/cars/1'
                elif self.prob_model.dataset_type == 'esfp':
                    save_dir = f'/home/user/app/tmp/poisson/esfp/1'
                elif self.prob_model.dataset_type == 'rvt':
                    save_dir = f'/home/user/app/tmp/poisson/gen1/1'
                else:
                    save_dir = f'/home/user/app/tmp/poisson/baloon_popping/1'

                # To compute time comment this
                #self.save_pdf(self.score_matrix[p], save_dir, p=p)

        # Determine score for this event
        score = self.score_matrix[p][pos_int[0], pos_int[1]]
        if self.flag_decay:
            score *= (1 - np.exp(- 150 * self.t_diff[p][pos_int]))  # if AT = 4 --> this is almost 0 already  (Caveat: this value is for seconds in the CALTECH101 dataset)
        #score_percentile = self.percentile_matrix[p][pos_int[0], pos_int[1]]

        # Determine acceptance probability and make decision
        prob_ev = min(self.prob_max, score)
        #prob_ev = min(self.prob_max, score_percentile)
        return random.random() < prob_ev
        #return self.filter_event(prob_ev)  --> 0.08 ms/kev of overhead time (800ms for 1e6 events)


class EventSamplerDeterministic(EventSampler):
    """Deterministically sample events based on a temporal window."""

    def __init__(self, **kwargs):
        super().__init__(prob_model=ProbIdentity(), **kwargs)
        
        # Set time scale factor based on dataset type
        self.t_scale_factor = 1e3 if config['dataset_type'] in ['timelens', 'rvt', 'esfp'] else 1e-3
        
        # Initialize temporal window parameters
        self.window_size = self.t_scale_factor * 0.1  # Window size of 0.1 ms by default
        self.t_init = np.zeros(2, dtype=float)
        self.t_window_in = np.full(2, self.sampling_rate * self.window_size)  # Acceptance of alpha * window size (e.g. alpha = 0.1, accept events from 0-0.01ms, reject from 0.01-0.1ms)

    def __call__(self, events, **kwargs):
        return self.sample_events(events)

    def eval_event(self, pos_int, t, p=0):
        """Evaluate whether to accept or reject an event based on its timestamp."""
        
        # Update window start time if necessary
        #if t >= self.t_init[p] + self.window_size:
        #    self.t_init[p] = t
        
        #res = self.t_init[p] <= t < (self.t_init[p] + self.t_window_in[p])

        # Accept event if it falls within the sampling window
        #return (t % self.window_size) < self.t_window_in[p]
        # Compute the time difference relative to the last window start
        time_diff = t - self.t_init[p]

        # If `t` has moved beyond the current window, update `t_init[p]`
        if time_diff >= self.window_size:
            self.t_init[p] += (time_diff // self.window_size) * self.window_size  # Jump to the latest valid window start

        # Check if `t` falls within the acceptance range
        return (t - self.t_init[p]) < self.t_window_in[p]


class EventSamplerUniform(EventSamplerProbabilistic):
    """Uniform event sampler that incorporates a prior probability."""

    def __init__(self, **kwargs):
        # Use ProbIdentity as the probability model for uniform sampling
        prob_model = ProbIdentity()
        super().__init__(prob_model=prob_model, **kwargs)

    def __call__(self, events, **kwargs):
        """Sample events uniformly."""
        return self.sample_events(events)

    def eval_event(self, pos_int, t, p=0):
        """
        Evaluate whether to accept or reject an event.
        
        Args:
            pos_int (tuple): Integer coordinates of the event.
            t (float): Timestamp of the event.
            p (int): Polarity of the event (0 or 1).
        
        Returns:
            bool: True if the event is accepted, False otherwise.
        """
        
        # Make decision based on calculated probability
        return random.random() < self.sampling_rate  # Check overhead time  --> 0.08 ms/kev of overhead time (800ms for 1e6 events)
        #return self.filter_event(self.sampling_rate)


class EventSamplerAccumGrid(EventSamplerProbabilistic):
    """Event sampler using an accumulating grid probability model."""

    def __init__(self, sensor_size, **kwargs):
        # Get coarse grid size, defaulting to 16x16 if not provided
        coarse_size = kwargs.get('coarse_size', (16, 16))
        
        # Initialize the accumulating grid probability model
        prob_model = AccumGridProb(grid_size=sensor_size, **kwargs)
        
        # Initialize the parent class
        super().__init__(sensor_size=sensor_size, prob_model=prob_model, **kwargs)
        
        # Get dimensions of the original and coarse grids
        h_orig, w_orig = prob_model.E_weights[0].shape
        h_coarse, w_coarse = prob_model.E_coarse[0].shape
        
        # Calculate the PDF scale factor
        # This scales the PDF mean to the desired sampling percentage
        self.pdf_scale_factor = h_orig * w_orig * self.sampling_rate

    def __call__(self, events, **kwargs):
        """Sample events using the accumulating grid method."""
        return self.sample_events(events)


class EventSamplerPoisson(EventSamplerProbabilistic):
    """Event sampler using a Poisson probability model."""

    def __init__(self, sensor_size, **kwargs):
        # Initialize the Poisson probability model
        prob_model = PoissonProb(grid_size=sensor_size, prob=config['prob_init'], **kwargs)
        
        # Initialize the parent class
        super().__init__(sensor_size=sensor_size, prob_model=prob_model, **kwargs)
        
        # Calculate the PDF scale factor
        h_orig, w_orig = prob_model.E_weights[0].shape
        self.pdf_scale_factor = h_orig * w_orig * self.sampling_rate
        
    def __call__(self, events, **kwargs):
        """Sample events using the Poisson method."""
        return self.sample_events(events)


class EventSamplerEvDownsamplingNavi(EventSampler):
    '''
    Implementation of "EvDownsampling" presented in Navi workshop at ECCV 2024.
    Project: https://sussex.figshare.com/articles/dataset/EvDownsampling_dataset/26528146
    Paper: https://drive.google.com/file/d/1s40YRb1HdJ7GMWotIpakDeKl9ETv8dd6/view 
    Github: https://github.com/anindyaghosh/EvDownsampling
    '''
    def __init__(self, **kwargs):
        super().__init__(prob_model=ProbIdentity(), **kwargs)
        
        # Set time scale factor based on dataset type (this sampler assumes input in microseconds)
        self.t_scale_factor = 1 if config['dataset_type'] in ['timelens', 'rvt', 'esfp'] else 1e6
        
        # Initialize temporal window parameters
        self.window_size = kwargs.get('window_size', 10) * self.t_scale_factor
        self.t_init = np.zeros(2, dtype=float)

        # Vars (from the original code)
        # Rules to keep more or less the desired sampling rate
        var_list = [2, 1, 420, 16, 1.2]

        # With dt=0.1 works a little bit better, but it's terribly slow
        if self.sampling_rate <= 0.05:
            var_list = [3, 1, 420, 16, 1.2]
        elif self.sampling_rate <= 0.1:
            var_list = [2, 1, 420, 16, 1.2]
        elif self.sampling_rate <= 0.2:
            var_list = [1, 1, 50, 10, 1.2]
        elif self.sampling_rate <= 0.5:
            var_list = [1, 1, 5, 2, 1.2]
        elif self.sampling_rate <= 0.9:
            var_list = [1, 1, 1, 1, 1]

        self.spatial_scale, self.dt, self.tau_theta, self.tau_accumulator, self.beta = var_list

    def __call__(self, events, **kwargs):
        return self.sample_events(events)

    def rescale_to_original_size(self, downsampled_events):
        """Rescale events to the original size."""
        # Define the new dtype where 't' is now a float
        new_dtype = np.dtype([
            ('x', '<i2'), 
            ('y', '<i2'), 
            ('t', '<f8'),  # Use float64 instead of int
            ('p', '<i1')
        ])

        # Create an empty structured array with the new dtype
        res_evs = np.empty(downsampled_events.shape, dtype=new_dtype)

        # Copy the existing integer values directly
        res_evs['x'] = downsampled_events['x'] * self.spatial_scale
        res_evs['y'] = downsampled_events['y'] * self.spatial_scale
        res_evs['p'] = downsampled_events['p']

        # Convert 't' to float and scale it
        res_evs['t'] = downsampled_events['t'].astype(np.float64) / self.t_scale_factor

        return res_evs

    def sample_events(self, events):
        if len(events['t']) == 0:
            return events
        
        x, y, t, p = events["x"], events["y"], events["t"], events["p"]
        x, y, t, p = filter_events_to_img(self.grid_size, x, y, t, p)
        events_filtered ={ "x": x, "y": y, "t": t, "p": p }

        dtype = np.dtype([('x', '<i2'), ('y', '<i2'), ('t', '<i8'), ('p', '<i1')])

        # Convert columns to match the structured dtype
        structured_array = np.array(
            list(zip(
                events_filtered['x'].astype(np.int16),
                events_filtered['y'].astype(np.int16),
                (events_filtered['t'] * self.t_scale_factor).astype(np.int64),  # Convert `t` to integer microseconds
                events_filtered['p'].astype(np.int8)
            )),
            dtype=dtype
        )

        downsampled_events = evDownNavi.event_downsample(events=structured_array, 
                                      sensor_size=(self.grid_shape[2], self.grid_shape[1], 2), 
                                      target_size=(int(math.ceil(self.grid_shape[2]/self.spatial_scale)), int(math.ceil(self.grid_shape[1]/self.spatial_scale))), 
                                      dt=self.dt, 
                                      tau_theta=self.tau_theta, 
                                      tau_accumulator=self.tau_accumulator, 
                                      beta=self.beta)

        # Initialize dictionary for new events
        new_events_dict = {"x": [], "y": [], "t": [], "p": []}
        
        # Rescale and filter
        downsampled_events_rescaled = self.rescale_to_original_size(downsampled_events)
        x, y, t, p = downsampled_events_rescaled["x"], downsampled_events_rescaled["y"], downsampled_events_rescaled["t"], downsampled_events_rescaled["p"]
        x, y, t, p = filter_events_to_img(self.grid_size, x, y, t, p)

        # Find the number of original events that are ≤ t
        p_list = [0, 1]  # Generic form
        mask_p0 = events_filtered['p'] == p_list[0]
        mask_p1 = events_filtered['p'] == p_list[1]

        if not self.sep_pols:
            mask_p0 = mask_p1 = [True] * len(events_filtered['p'])

        num_evs_per_t = {
            int(p_list[0]): np.searchsorted(events_filtered['t'][mask_p0], t, side='right') + self.num_evs_orig[p_list[0]],
            int(p_list[1]): np.searchsorted(events_filtered['t'][mask_p1], t, side='right') + self.num_evs_orig[p_list[1]]
        }

        # To boost the results, shuffle the indeces over same "t" values, because if not capping the amount of events prioritizes lower x,y and p=0.
        # This is a way to avoid this bias.
        # Group indices by unique t values
        t_bins = defaultdict(list)
        for i, t_val in enumerate(t):
            t_bins[t_val].append(i)

        # Shuffle indices within each t bin
        shuffled_indices = []
        for t_val, indices in t_bins.items():
            np.random.shuffle(indices)
            shuffled_indices.extend(indices)  # Preserve global t order

        #for i in range(len(t)):
        for i in shuffled_indices:
            x_, y_, t_, p_ = x[i], y[i], t[i], p[i]
            p_int_ = int(p_)

            if not self.sep_pols:
                p_int_ = 0  # Don't separate polarities

            is_accept = True  # Already downsampled

            # Compute relative sampling ratio
            #self.rel_original_samples = self.num_evs_smpl[p_int_] / num_evs_per_t[p_int_][i]
            self.rel_original_samples = np.sum(self.num_evs_smpl) / (num_evs_per_t[p_list[0]][i] + num_evs_per_t[p_list[1]][i])

            if self.rel_original_samples > self.sampling_rate:
                is_accept = False

            if is_accept:
                self.num_evs_smpl[p_int_] += 1  # Update sampled count
                new_events_dict["x"].append(x_)
                new_events_dict["y"].append(y_)
                new_events_dict["t"].append(t_)
                new_events_dict["p"].append(p_)

        # Store the updated event count back
        self.num_evs_orig[p_list[0]] = num_evs_per_t[p_list[0]][-1]
        self.num_evs_orig[p_list[1]] = num_evs_per_t[p_list[1]][-1]

        # Convert lists to numpy arrays while preserving original data types
        for key in new_events_dict:
            new_events_dict[key] = np.array(new_events_dict[key], dtype=events_filtered[key].dtype)

        return new_events_dict



def main(event_processor, ev_save_path, **kwargs):
    """Main function to process and sample events."""
    # Set random seeds for reproducibility (each process spawns its own random seed)
    np.random.seed(int(config['num_exp']))
    random.seed(int(config['num_exp']))   

    # Initialize event sampler based on SAMPLER_TYPE
    event_sampler = initialize_event_sampler(event_processor.size_y, event_processor.size_x)
    
    # Update priors in case it's needed
    new_prior = kwargs.get('prior', None)
    if new_prior is not None:
        event_sampler.update_prior(new_prior)
    
    # Initialize event file saver
    event_file_saver = initialize_event_file_saver(event_processor, ev_save_path)
    
    # Process events
    process_events(event_processor, event_sampler, event_file_saver, ev_save_path)

def initialize_event_sampler(size_y, size_x):
    """Initialize and return the appropriate event sampler based on SAMPLER_TYPE."""
    sensor_size = [size_y, size_x]
    sampler_classes = {
        'uniform': EventSamplerUniform,
        'det': EventSamplerDeterministic,
        'accum': EventSamplerAccumGrid,
        'poisson': EventSamplerPoisson,
        'evDownNavi': EventSamplerEvDownsamplingNavi
    }
    
    sampler_class = sampler_classes.get(config['sampler_type'])
    if not sampler_class:
        raise ValueError(f"{config['sampler_type']} is not available.")
    
    return sampler_class(sensor_size=sensor_size, prob_init=config['prob_init'], **get_sampler_kwargs())

def get_sampler_kwargs():
    """Return a dictionary of keyword arguments for the sampler based on SAMPLER_TYPE."""
    common_kwargs = {
        'alpha_prior': config['alpha_prior'],
        'coarse_size': (config['coarse_size'], config['coarse_size']),
        't_window_size': config['t_window_size'],
        'decay': config['decay'],
        'alpha': config['alpha'],
        'dataset_type': config['dataset_type'],
        'sep_pols': config['sep_pols'],
        'flag_t_surfaces': config['t_surfaces'],
        'debug': config['debug']
    }
    
    sampler_specific_kwargs = {
        'det': common_kwargs,
        'uniform': common_kwargs,
        'accum': common_kwargs,
        'poisson': common_kwargs,
        'evDownNavi': common_kwargs
    }
    
    return sampler_specific_kwargs.get(config['sampler_type'], {})

def initialize_event_file_saver(event_processor, ev_save_path):
    """Initialize and return an EventFileSaver object."""
    save_path = ev_save_path if ev_save_path.is_dir() else Path(ev_save_path.parent)
    chunks = event_processor.chunks if event_processor.evs_dataset_type in ['rvt', 'esfp'] else -1
    max_len = event_processor.h5_len if event_processor.evs_dataset_type in ['rvt', 'esfp'] else 0
    divider = event_processor.divider if event_processor.evs_dataset_type == 'rvt' else -1
    
    return EventFileSaver(evs_write_path=save_path, dataset_type=event_processor.evs_dataset_type,
                          size_y=event_processor.size_y, size_x=event_processor.size_x,
                          maxshape=max_len, divider=divider, chunks=chunks)

def process_events(event_processor, event_sampler, event_file_saver, ev_save_path):
    """Process events using the given event processor, sampler, and saver."""
    i = 0
    num_evs_total = 0
    total_time = 0
    
    while True:
        if config['debug']:
            print(i, end="\r")
        
        evs = event_processor.read_next_event_packets(num_packets=1)
        if evs == -1:
            break
        
        num_evs_total += len(evs['t'])
        
        evs_filename = get_evs_filename(event_processor, ev_save_path, i)
        
        start_time = time.perf_counter()
        evs_sampled = event_sampler(evs)
        total_time += (time.perf_counter() - start_time)
        
        event_file_saver.save_sampled_events(evs_sampled, evs_filename=evs_filename)
        i += 1
        
        if config['debug'] and i > 1000:
            break
    
    if config['debug']:
        print_debug_info(num_evs_total, total_time, time_name="total_time")
        print_debug_info(num_evs_total, event_sampler.time_pdf_est_, time_name="time_pdf_est")
        print_debug_info(num_evs_total, event_sampler.time_eval_, time_name="time_eval")

def get_evs_filename(event_processor, ev_save_path, i):
    """Get the appropriate filename for saving events based on the dataset type."""
    if event_processor.evs_dataset_type == 'timelens':
        return event_processor.ev_files_list[i].name
    elif event_processor.evs_dataset_type == 'caltech101rpg':
        return event_processor.events_file.name
    elif event_processor.evs_dataset_type == 'esfp':
        return ev_save_path / "events.h5"
    else:
        return ev_save_path.name

def print_debug_info(num_evs_total, total_time, time_name="total_time"):
    """Print debug information about the sampling process."""
    kevs = num_evs_total / 1000
    total_time_ms = round(total_time * 1000)
    print(f"Sampler: {config['sampler_type']}, {time_name}: {total_time}, ms/Kev: {round(total_time_ms / kevs, 2)}, Events: {num_evs_total}, "
          f"t_window: {config['t_window_size']}")

def main_caltech(ev_path, ev_save_path):
    """Main function for processing Caltech101 dataset."""
    event_processor = EventProcessorCaltech101RPG(evs_file=ev_path, bin_size=1000)
    ev_save_path = ev_save_path / f"events_{config['sampler_type']}_{config['prob_init']}-{config['num_exp']}"
    print(f"Saving new events to {ev_save_path}")
    main(event_processor, ev_save_path)
    print(f"Finished {ev_path.name}!")

def main_caltech_mp(tuples_list):
    """Multiprocessing function for Caltech101 dataset."""
    evs_file, ev_save_path = tuples_list
    print(f"Processing: {evs_file.stem}")
    event_processor = EventProcessorCaltech101RPG(evs_file=evs_file, bin_size=10000)
    main(event_processor, ev_save_path)

def process_sequence(seq_path):
    """Process a sequence of events based on the dataset type."""
    if config['dataset_type'] == 'timelens':
        process_timelens_sequence(seq_path)
    elif config['dataset_type'] in ['caltech101', 'cars']:
        process_caltech_cars_sequence(seq_path)
    elif config['dataset_type'] == 'rvt':
        process_rvt_sequence(seq_path)
    elif config['dataset_type'] == 'esfp':
        process_esfp_sequence(seq_path)
    else:
        print(f"{config['dataset_type']} is not available.")

def process_timelens_sequence(seq_path):
    """Process a Timelens dataset sequence."""
    ev_path = seq_path / "events" / "original"
    ev_save_path = config['ev_save_path'] / seq_path.stem / "events" / f"{config['exp_name']}" / f"events_{config['sampler_type']}_{config['prob_init']}-{config['num_exp']}"
    img_path = seq_path / "images"
    tstmp_path = seq_path / "images"
    
    event_processor = EventProcessorTimelens(evs_path=ev_path, imgs_path=img_path, tstmp_path=tstmp_path)
    
    print(f"Saving new events to {ev_save_path}")
    ev_save_path.mkdir(parents=True, exist_ok=True)
    os.chmod(ev_save_path, 0o777)
    main(event_processor, ev_save_path)
    print(f"Finished {seq_path.name}!")

def process_caltech_cars_sequence(seq_path):
    """Process a Caltech101 or N-cars dataset sequence."""
    evs_file_list = [ev_path for ev_path in seq_path.iterdir() if ev_path.is_file()]
    
    if config['dataset_type'] == 'caltech101':
        ev_save_path = config['ev_save_path'] / f"events_{config['sampler_type']}_{config['prob_init']}-{config['num_exp']}" / f"{config['split']}" / seq_path.name
    else:
        ev_save_path = config['ev_save_path'] / f"events_{config['sampler_type']}_{config['prob_init']}-{config['num_exp']}" / ("train" if config['split'] == 'training' else "test") / seq_path.name
    
    print(f"Saving new events to {ev_save_path}")
    ev_save_path.mkdir(parents=True, exist_ok=True)
    os.chmod(ev_save_path, 0o777)
    
    num_proc2 = min(config['num_proc'], len(evs_file_list))
    tuples_list = [(evs_file, ev_save_path) for evs_file in evs_file_list]
    with multiprocessing.Pool(processes=num_proc2) as pool2:
        pool2.map(main_caltech_mp, tuples_list)
    
    print(f"Finished {seq_path.name}!")

def process_rvt_sequence(seq_path):
    """Process an RVT dataset sequence."""
    lab_orig_path = seq_path.parent / Path(seq_path.stem[:-7] + '_bbox').with_suffix('.npy')
    mode_rvt = 'train' if config['split'] == 'training' else ('val' if config['split'] == 'validation' else 'test')
    
    ev_save_path = config['ev_save_path'] / f"events_{config['sampler_type']}_{config['prob_init']}-{config['num_exp']}" / f"{mode_rvt}" / seq_path.name
    lab_save_path = config['ev_save_path'] / f"events_{config['sampler_type']}_{config['prob_init']}-{config['num_exp']}" / f"{mode_rvt}" / lab_orig_path.name
    
    event_processor = EventProcessorRVT(evs_path=seq_path)
    ev_save_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(ev_save_path.parent, 0o777)
    
    print(f"Saving new events to {ev_save_path}")
    main(event_processor, ev_save_path)
    print(f"Finished {seq_path.name}!")
    
    # Relative path for the label file to work from docker or outside
    rel_lab_path = os.path.relpath(lab_orig_path, lab_save_path.parent)
    if not lab_save_path.exists():
        lab_save_path.symlink_to(rel_lab_path)


def process_esfp_sequence(seq_path):
    """Process an ESFP dataset sequence."""
    ev_save_path = config['ev_save_path'] / f"events_{config['sampler_type']}_{config['prob_init']}-{config['num_exp']}" / f"{config['split']}" / seq_path.name

    evs_path = seq_path / "events.h5"
    event_processor = EventProcessorESFP(evs_path=evs_path)
    ev_save_path.mkdir(parents=True, exist_ok=True)
    assert(ev_save_path.is_dir())
    os.chmod(ev_save_path, 0o777)
    
    # Prior from annotations (phone)
    prior = None
    if config['prior'] is not None:
        annot_path = seq_path / f"{seq_path.name}.json"
        if annot_path.exists():
            phone_mask = ut.load_masks_json(annot_path, labels=['phone'])
            # Inverse of the phone mask (to avoid the phone)
            prior_phone = 1 - phone_mask
            object_mask = ut.load_masks_json(annot_path, labels=['object'])
            # Intersection of the phone and object masks (some object has phone inside the bbox), convert to binary mask
            #prior = np.clip(prior_phone * object_mask, 0, 1)
            #prior = object_mask
            #prior = prior_phone

            prior = None  # Just go with no prior

    print(f"Saving new events to {ev_save_path}")
    main(event_processor, ev_save_path, prior=prior)
    print(f"Finished {seq_path.name}!")

    # Relative path for the gt files to work from docker or outside
    other_files = [f for f in seq_path.iterdir() if f.is_file() and f.name != "events.h5"]
    
    for f in other_files:
        rel_f_path = os.path.relpath(f, ev_save_path)
        if (ev_save_path / f.name).exists():
            # Delete if it already exists and do it again
            #print(f"Deleting {ev_save_path / f.name}")
            try:
                (ev_save_path / f.name).unlink()
                #print("File deleted successfully.")
            except FileNotFoundError:
                print(f"{ev_save_path / f.name} File not found.")
            except PermissionError:
                print(f"{ev_save_path / f.name}Permission denied.")
            except Exception as e:
                print(f"{ev_save_path / f.name} An error occurred: {e}")

        (ev_save_path / f.name).symlink_to(rel_f_path)


def process_sequences(config):
    """Process multiple sequences based on the dataset type."""
    seqs_path = Path(config['seqs_path'])
    
    # Get sequence paths based on dataset type
    if config['dataset_type'] == 'rvt':
        seq_paths = sorted([seq_path for seq_path in seqs_path.iterdir() if seq_path.is_file() and seq_path.suffix == '.h5'])
    else:
        seq_paths = sorted([seq_path for seq_path in seqs_path.iterdir() if seq_path.is_dir()])

    # Process sequences
    num_proc = min(config['num_proc'], len(seq_paths))
    if config['debug']:
        process_debug_sequence(config, seq_paths)
    elif config['dataset_type'] in ['caltech101', 'cars']:
        for seq_path in seq_paths:
            process_sequence(seq_path)
    else:
        with multiprocessing.Pool(processes=num_proc) as pool:
            pool.map(process_sequence, seq_paths)

def process_debug_sequence(config, seq_paths):
    """Process a single sequence for debugging."""
    if config['dataset_type'] == 'rvt':
        process_sequence(seq_paths[116])
    else:
        #process_sequence(seq_paths[19])
        process_sequence(seq_paths[-5])
        #process_sequence(seq_paths[0])

def process_single_file(config):
    """Process a single file (for Caltech dataset)."""
    ev_path = Path(config['ev_path'])
    ev_save_path = Path(config['ev_save_path'])
    main_caltech(ev_path, ev_save_path)



if __name__ == "__main__":
    # Run example:
    # python event_sampling --dataset_name NCaltech101 --sampler poisson --prob_init 1 --d

    # Parse command line arguments
    print(config)

    # Process sequences
    if config['seqs_path']:
        multiprocessing.set_start_method('spawn', force=True)
        process_sequences(config)
    else:
        process_single_file(config)

    exit()

