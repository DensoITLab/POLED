# All the probability function definitions will be here
import numpy as np
import scipy
from scipy.ndimage import convolve
from scipy.special import factorial
import pandas as pd

import utils as ut


class ProbFunction:
    def __init__(self, *args, **kwargs):
        # Variables
        self.dataset_type = kwargs.get('dataset_type', None)
        # Prob from event downsampling for background (test)
        if self.dataset_type == 'cars':
        # if self.dataset_type == 'cars' or self.dataset_type == 'caltech101':
            # N-Cars dataset events are too sparse in a very small sensor
            self.background_prob = 0
        else:
            # self.background_prob = kwargs.get('prob', 1)
            self.background_prob = 0
        # Debug vars
        self.flag_debug = kwargs.get('debug', False)

    @staticmethod
    def get_time_diff(t_mat, pos, t):
        # Get time difference with respect to previous event in that same position of the grid
        return (t - t_mat[pos]) * 1e6  # To miliseconds -> caveat with timelens dataset, this is for Caltech and Cars

    def __call__(self, *args, **kwargs):
        # To be defined in every child class
        pass


###+ Probability Parent Classes +###

class ProbIdentity(ProbFunction):
    def __init__(self, **kwargs):
        super(ProbIdentity, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return 0


class AccumGridProbFunction(ProbFunction):
    # Define probability function based on a spatial grid
    def __init__(self, grid_size, *args, **kwargs):
        ProbFunction.__init__(self, *args, **kwargs)
        # Variables
        self.t_window_size = kwargs.get('t_window_size', 10)  # In ms
        self.E_weights = None
        self.E_accum = None  # Event accumulator matrix
        self.E_coarse = None  # Coarse Event accumulator matrix
        self.t_current = np.array([0, 0], dtype=float)
        self.t_init = np.array([0, 0], dtype=float)
        self.T_accum = {}  # In the form of a dictionary
        if self.dataset_type == 'timelens' or self.dataset_type == 'rvt' or self.dataset_type == 'ramnet' or self.dataset_type == 'esfp':
            self.t_window = np.array([self.t_window_size * 1e3, self.t_window_size * 1e3], dtype=float)  # (micro) Temporal window size (maybe make adaptive later)  --> timelens
            self.alpha = kwargs.get('alpha', 1)

        else:
            self.t_window = np.array([self.t_window_size * 1e-3, self.t_window_size * 1e-3], dtype=float)  # (seconds) Temporal window size (maybe make adaptive later)  --> caltech
            self.alpha = kwargs.get('alpha', 1)
            self.tdiff_scaling = 1e6
            
        self.grid_prob_size = grid_size
        self.grid_prob_coarse_size = kwargs.get('coarse_size', (16, 16))
        self.flag_t_surfaces = kwargs.get('flag_t_surfaces', False)
        self.flag_weights_init = False
        self.eps = 1e-10
        # For debugging
        self.idx_ = 0
        # Initialize variables
        self._init_grids_()

    def _init_grids_(self):
        self.E_weights = np.zeros((2, self.grid_prob_size[0] + 1, self.grid_prob_size[1] + 1), dtype=float)
        self.E_accum = np.ones((2, self.grid_prob_size[0] + 1, self.grid_prob_size[1] + 1), dtype=float) * self.background_prob
        h_coarse = int(self.grid_prob_size[0] / self.grid_prob_coarse_size[0]) + 1
        w_coarse = int(self.grid_prob_size[1] / self.grid_prob_coarse_size[1]) + 1
        self.E_coarse = np.ones((2, h_coarse, w_coarse), dtype=float)
        # Event time surface initialization (2 polarities)
        if self.flag_t_surfaces:
            self.set_init_tsurface(0)
            self.set_init_tsurface(1)

    def set_init_tsurface(self, p):
        self.T_accum[p] = {}
        for y in range(self.grid_prob_size[0]):
            self.T_accum[p][y] = {}
            for x in range(self.grid_prob_size[1]):
                self.T_accum[p][y][x] = []

    def set_init_time(self, t_init, p=0):
        self.t_init[p] = t_init

    def acc_event(self, pos_int, t, p):
        if self.flag_t_surfaces:
            self.T_accum[p][pos_int[0]][pos_int[1]].append(t)
        else:
            self.E_accum[p][pos_int] += 1

        self.t_current[p] = t

    def set_values_end_window(self, p):
        # Reset accum and coarse grids
        self.E_accum[p] = 1 * self.background_prob
        self.E_coarse[p] = 1
        # Advance the temporal window starting point
        self.t_init[p] += self.t_window[p]
        # Time surfaces
        if self.flag_t_surfaces:
            self.set_init_tsurface(p)

    def compute_sum_ev_times_decay(self, p):
        # Compute the time surface
        E_accum_t_ = np.zeros((self.grid_prob_size[0] + 1, self.grid_prob_size[1] + 1), dtype=float)
        thau = 1
        for y in range(self.grid_prob_size[0]):
            for x in range(self.grid_prob_size[1]):
                E_accum_t_[y, x] = np.sum([np.exp(-(self.t_current[p] - self.T_accum[p][y][x][i])/(thau*self.t_window[p])) for i in range(len(self.T_accum[p][y][x]))])

        return E_accum_t_ + self.background_prob  # Init with background prob

    def compute_coarse_accum(self, p, t=0):
        # Compute weights of coarse event window for a certain polarity
        # E_accum_w_ = self.E_accum[p]
        # E_accum_tsf_w_ = self.E_accum_tsurface[p]
        # Time surfaces
        if self.flag_t_surfaces:
            E_accum_w_ = self.compute_sum_ev_times_decay(p)
        else:
            E_accum_w_ = self.E_accum[p]

        E_coarse_w_ = np.zeros_like(self.E_coarse[p])  # Initialize E_coarse as for the window
        y_orig, x_orig = E_accum_w_.shape
        y_coarse, x_coarse = E_coarse_w_.shape

        sub_rows = (y_orig // y_coarse) + 1
        sub_cols = (x_orig // x_coarse) + 1
        for i in range(y_coarse):
            for j in range(x_coarse):
                start_row = i * sub_rows
                start_col = j * sub_cols
                E_coarse_w_[i, j] = E_accum_w_[start_row:start_row+sub_rows, start_col:start_col+sub_cols].sum()  # Compute the sum of elements from self.E_accum
                # Test time surfaces
                # E_coarse_w_[i, j] = np.exp(-t/self.t_window[p]) * E_accum_tsf_w_[start_row:start_row + sub_rows, start_col:start_col + sub_cols].sum()
                # Time surfaces
                # E_coarse_w_[i, j] = E_accum_tsf_w_[start_row:start_row + sub_rows, start_col:start_col + sub_cols].sum()

        return E_coarse_w_

    def compute_coarse_weights(self, p, t=0):
        E_coarse_w_ = self.compute_coarse_accum(p, t)

        # TODO: Normalize properly
        # Testing normalizations
        # return np.exp(E_coarse_w_)/np.exp(E_coarse_w_).sum()
        # Return normalized weights
        # E_coarse_w_ += 1  # To not to have a lot of 0s
        return E_coarse_w_/(E_coarse_w_.sum() + self.eps)

    def resize_coarse_to_orig(self, coarse_mat):
        # Compute weights of coarse event window for a certain polarity
        E_orig_ = np.zeros_like(self.E_accum[0], dtype=float) * coarse_mat.min()  # Initialize E_orig as for the window
        y_orig, x_orig = E_orig_.shape
        y_coarse, x_coarse = coarse_mat.shape

        sub_rows = (y_orig // y_coarse) + 1
        sub_cols = (x_orig // x_coarse) + 1
        for i in range(y_coarse):
            for j in range(x_coarse):
                start_row = i * sub_rows
                start_col = j * sub_cols
                E_orig_[start_row:start_row+sub_rows, start_col:start_col+sub_cols] = coarse_mat[i, j]

        # TODO: fill 0s with some mirroring technique or sth
        return E_orig_/(E_orig_.sum() + self.eps)

    def combine_prev_weights(self, weights_mat, p):
        # TODO: Improve name and scope
        # Function to combine current weight_matrix with previous ones
        # alpha = 1  # If not initialized just assign all the weight to the weight matrix
        alpha = self.alpha if self.flag_weights_init else 1  # If not initialized just assign all the weight to the weight matrix
        weights_mat_new = (weights_mat * alpha) + (self.E_weights[p] * (1-alpha))

        self.flag_weights_init = True
        return weights_mat_new


class PoissonProbFunction(AccumGridProbFunction):
    # Define probability function based on a spatial grid
    def __init__(self, grid_size, *args, **kwargs):
        ProbFunction.__init__(self, *args, **kwargs)
        # Variables
        self.t_window_size = kwargs.get('t_window_size', 10)  # In ms
        self.E_weights = None
        self.E_accum = None  # Event accumulator matrix
        self.df_vals = []
        self.t_current = np.array([0, 0], dtype=float)
        self.t_init = np.array([0, 0], dtype=float)
        self.T_accum = {}  # In the form of a dictionary
        if self.dataset_type == 'timelens' or self.dataset_type == 'rvt' or self.dataset_type == 'ramnet' or self.dataset_type == 'esfp':
            self.t_window = np.array([self.t_window_size * 1e3, self.t_window_size * 1e3], dtype=float)  # (micro) Temporal window size (maybe make adaptive later)  --> timelens
            self.alpha = kwargs.get('alpha', 1)

        else:
            self.t_window = np.array([self.t_window_size * 1e-3, self.t_window_size * 1e-3], dtype=float)  # (seconds) Temporal window size (maybe make adaptive later)  --> caltech
            self.alpha = kwargs.get('alpha', 1)
            self.tdiff_scaling = 1e6
            
        self.grid_prob_size = grid_size
        self.grid_prob_coarse_size = kwargs.get('coarse_size', (16, 16))
        self.flag_t_surfaces = kwargs.get('flag_t_surfaces', False)
        self.flag_weights_init = False
        self.eps = 1e-10
        self.df = pd.DataFrame(columns=['x', 'y', 't', 'p'])
        # For debugging
        self.idx_ = 0
        # Initialize variables
        self._init_grids_()

    def acc_event(self, pos_int, t, p):
        self.E_accum[p][pos_int] += 1
        self.t_current[p] = t
        self.df_vals.append([pos_int[1], pos_int[0], t, p])
        #self.df = pd.concat([self.df, pd.DataFrame([pos_int[1], pos_int[0], t,  p])], ignore_index=True)
        #df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    def resize_coarse_to_orig(self, coarse_mat):
        # Compute weights of coarse event window for a certain polarity
        E_orig_ = np.zeros_like(self.E_accum[0], dtype=float) * coarse_mat.min()  # Initialize E_orig as for the window
        y_orig, x_orig = E_orig_.shape
        y_coarse, x_coarse = coarse_mat.shape

        sub_rows = (y_orig // y_coarse) + 1
        sub_cols = (x_orig // x_coarse) + 1
        for i in range(y_coarse):
            for j in range(x_coarse):
                start_row = i * sub_rows
                start_col = j * sub_cols
                E_orig_[start_row:start_row+sub_rows, start_col:start_col+sub_cols] = coarse_mat[i, j]

        return E_orig_

    def compute_k(self, p):
        self.df = pd.DataFrame(self.df_vals, columns=['x', 'y', 't', 'p'])
        df_p = self.df[self.df['p'] == p]
        df_diff = ut.compute_evs_tdiff(df_p)
        df_diff_filtered = ut.filter_evs_tdiff(df_diff, xstart=0.002)   # Bug: xstart should depend on the dataset
        tdiff_hist, bins = ut.compute_tdiff_hist(df_diff_filtered)
        # Calculate the tdiff corresponding to the average point of the histogram
        # Compute the midpoints of the bins
        bin_midpoints = (bins[:-1] + bins[1:]) / 2
        # Calculate the mean (expected value)
        mean_tdiff = np.sum(bin_midpoints * tdiff_hist) / (np.sum(tdiff_hist) + 1)

        k = self.t_window[p] / mean_tdiff

        return k
    
    def compute_poiss_probs(self, grid, k_list=None, sigma=0, lambda_smooth=False, log_lambda=False, suppression_strength=0.0, box_filter=(2, 2)):
        """
        Computes Poisson probabilities with spatial smoothing and density-dependent suppression.

        Parameters:
        - grid: 2D numpy array representing event intensities (lambda values).
        - k_list: List of k values to compute probabilities for. If None, defaults to [0].
        - sigma: Gaussian smoothing factor to reduce bias.
        - adaptive: If True, applies log1p to λ before computing probabilities.
        - suppression_strength: Controls how much clustering reduces probabilities (higher = more suppression).

        Returns:
        - grid_poi: Poisson probability heatmap with suppression.
        - density_map: The local density used for suppression.
        """
        if k_list is None:
            k_list = [0]  # Default to computing P(0)

        # Smooth λ values to handle local variations
        smoothed_lambda = grid

        if lambda_smooth:
            smoothed_lambda = scipy.ndimage.gaussian_filter(grid, sigma=sigma)

        # Compute density map (local event concentration)
        suppression_factor = 1.0

        if suppression_strength > 0:
            density_kernel = np.ones(box_filter)  # A 5x5 box filter for density estimation
            density_map = convolve(smoothed_lambda, density_kernel, mode='reflect')

            # Compute suppression factor (inverse density scaling)
            suppression_factor = 1 / (1 + (suppression_strength * density_map))

        # Adaptive Poisson Correction (Log transformation to reduce bias)
        if log_lambda:
            smoothed_lambda = np.log1p(smoothed_lambda)  # log(1 + x) ensures non-negative values

        # Compute Poisson probabilities
        grid_exp = np.exp(-smoothed_lambda)
        grid_poi = np.zeros_like(grid)

        for k in k_list:
            grid_poi += ((smoothed_lambda) ** k) * grid_exp / factorial(k)

        # Prob = 1 - P(0)
        prob_nonzero = 1 - grid_poi

        # Apply suppression to reduce probability in clustered areas
        prob_nonzero *= suppression_factor

        # Test
        grid_poi_pos = grid_poi

        return prob_nonzero, prob_nonzero
    
    def set_values_end_window(self, p):
        # TODO: rename this function and set its scope
        # Reset accum and coarse grids
        self.E_accum[p] = 1 * self.background_prob
        self.t_init[p] += self.t_window[p]
        self.df_vals = []

###- Probability Parent Classes -###

###+ Probability Instance Classes +###

class AccumGridProb(AccumGridProbFunction):
    def __init__(self, grid_size, *args, **kwargs):
        AccumGridProbFunction.__init__(self, grid_size, *args, **kwargs)
        # Variables
        self.flag_weights_update = False

    def eval_event(self, **kwargs):
        pos_ = kwargs.get("pos_int", None)
        t_ = kwargs.get("t", None)
        p_ = kwargs.get("p", None)
        flag_accumulate = kwargs.get("flag_accumulate", True)  # In case we want to
        self.flag_weights_update = False
        # Accumulate events
        if flag_accumulate:
            self.acc_event(pos_, t_, p_)
        if t_ > (self.t_init[p_] + self.t_window[p_]):
            # We updated the weights
            self.flag_weights_update = True
            weights_coarse = self.compute_coarse_weights(p_, t_)
            weights_mat = self.resize_coarse_to_orig(weights_coarse)
            self.E_weights[p_] = self.combine_prev_weights(weights_mat, p_)
            self.set_values_end_window(p_)

    def __call__(self, *args, **kwargs):
        self.eval_event(**kwargs)

class PoissonProb(PoissonProbFunction):
    def __init__(self, grid_size, *args, **kwargs):
        PoissonProbFunction.__init__(self, grid_size, *args, **kwargs)
        # Variables
        self.flag_weights_update = False

    def eval_event(self, **kwargs):
        pos_ = kwargs.get("pos_int", None)
        t_ = kwargs.get("t", None)
        p_ = kwargs.get("p", None)
        flag_accumulate = kwargs.get("flag_accumulate", True)
        self.flag_weights_update = False
        # Accumulate events
        if flag_accumulate:
            self.acc_event(pos_, t_, p_)
        if t_ > (self.t_init[p_] + self.t_window[p_]):
            # We updated the weights
            self.flag_weights_update = True
            grid = self.E_accum[p_]
            grid_poi, _ = self.compute_poiss_probs(grid, k_list=[0], sigma=1, lambda_smooth=False, log_lambda=False, suppression_strength=0, box_filter=(3, 3))  # NCaltech101, Gen1
 
            # Estimated PDF
            pdf_hat = grid_poi
            pdf_hat = scipy.ndimage.gaussian_filter(grid_poi, sigma=2)  # This seems to work consistently better for ESFP, NCaltech and RVT (didn't try the other method for Timelens) datasets than smoothing the lambda inside the poisson function

            # Normalize
            pdf_hat = pdf_hat / pdf_hat.sum()

            self.E_weights[p_] = pdf_hat
            self.set_values_end_window(p_)

    def __call__(self, *args, **kwargs):
        self.eval_event(**kwargs)


###+ Probability Instance Classes +###


def main():
    ev_k = [3, 2, 12*1e-6, 1]
    aa = KernelProbExpDec(kernel_size=3)
    t_mat = np.array([[0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 3, 10, 0, 0],
                      [0, 0, 0, 0, 0]]) * 1e-6
    ev_t = ev_k[2]
    ev_pos = [ev_k[0], ev_k[1]]
    print(aa(t_mat=t_mat, pos=ev_pos, t =ev_t))

    bb = ProbIdentity(kernel_size=3)
    t_c = 10
    t_p = 1
    t_diff = t_c - t_p
    print(bb(t=t_diff))


if __name__ == "__main__":
    main()
    print("Finished!")
