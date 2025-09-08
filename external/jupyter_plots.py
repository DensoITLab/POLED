from itertools import islice
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Function to resize a numpy array image given the size
def resize_image_array(image: np.ndarray, size: tuple) -> np.ndarray:
    return np.array(Image.fromarray(image).resize(size, Image.NEAREST))

# Function to scale the image value to [0, 255]
def normalize_scale_image(image: np.ndarray) -> np.ndarray:
    return (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)


def read_sequence_images(img_path: Path, **kwargs) -> list:
    # Read multiple frames from path
    n_frames = kwargs.get('n_frames', 1)
    start_frame = kwargs.get('start_frame', 0)
    step = kwargs.get('step', 1)
    im_size = kwargs.get('im_size', (420, 300))

    frames_path_list = list(sorted(img_path.iterdir()))  # Read all so we can skip
    frames_list = []

    for i in range(start_frame, start_frame + (n_frames * step), step):
        frame_ = Image.open(frames_path_list[i])
        frames_list.append(np.array(frame_.resize(im_size, Image.NEAREST)))

    return frames_list


def generate_evs_density_df(df_orig: pd.DataFrame, t_conv=1, t_res='10ms') -> pd.DataFrame:
    # Sort the DataFrame by time
    df = df_orig.sort_values('t')
    # Convert in case the datasets is in s/ms/us. Assume ms as default
    df['t_ms'] = df['t'] * t_conv
    # Erase time offset
    df['t_ms'] = df['t_ms'] - df.loc[0, 't_ms']
    # Set the time column as the index
    df.set_index(pd.to_datetime(df['t_ms'], unit='ms'), inplace=True)
    # Resample the DataFrame at 1 millisecond intervals and calculate the sum of events
    df_resampled = df['t'].astype(bool).resample(t_res).sum()
    # Forward fill missing values (events) and replace NaN with 0
    df_resampled = df_resampled.ffill().fillna(0)
    return df_resampled


def plot_evs_density(df_orig_list: list or pd.DataFrame, **kwargs) -> None:
    # Params
    t_conv = kwargs.get('t_conv', 1)
    t_res = kwargs.get('t_res', '10ms')

    if type(df_orig_list) != list:
        df_orig_list = [df_orig_list]

    df_plot_list = []
    for df_orig in df_orig_list:
        df_plot = generate_evs_density_df(df_orig, t_conv, t_res)
        df_plot_list.append(df_plot)

    # Plot the datasets
    fig, ax = plt.subplots(figsize=(10, 6))
    for df_plot in df_plot_list:
        plt.plot(df_plot.index, df_plot.values, marker='o')

    plt.xlabel('Time (ms)')
    plt.ylabel('Number of Events')
    plt.title('Number of Events over Time')

    # Rotate and align the x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def generate_hist_frames(hist_list, **kwargs):
    # Find global maximum values for consistent axes
    global_max_x = max(max(hist[1]) for hist in hist_list)
    global_max_y = max(max(hist[0]) for hist in hist_list)

    # Params
    global_max_x = kwargs.get('global_max_x', global_max_x)
    global_max_y = kwargs.get('global_max_y', global_max_y)
    id_list = kwargs.get('id_list', [None]*len(hist_list))
    fig_size = kwargs.get('fig_size', (10, 6))
    units = kwargs.get('units', ' ')
    xlabel = kwargs.get('xlabel', ' ')
    ylabel = kwargs.get('ylabel', ' ')
    title = kwargs.get('title', ' ')

    frames = []
    
    # Generate a plot for each temporal window
    for id_, hist_bins_ in zip(id_list, hist_list):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=fig_size)
        canvas = FigureCanvas(fig)  # Attach canvas to figure
    
        # Plot the histogram
        ax.plot(hist_bins_[1][:-1], hist_bins_[0], alpha=0.8)
        ax.set_xlim(0, global_max_x)
        ax.set_ylim(0, global_max_y)
    
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {id_} {units}")
        ax.grid(True)
    
        # Render the figure to a NumPy array
        canvas.draw()
        frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')  # Use buffer_rgba instead
        frame = frame.reshape(canvas.get_width_height()[::-1] + (4,))  # Convert to (H, W, 4) for RGBA
        frames.append(frame)
    
        # Close the figure to save memory
        plt.close(fig)
    
    return frames


# Resample event dataframe to fixed intervals to plot
def generate_plot_dataframe(df_orig, dataset_type='caltech101'):
    # Sort the DataFrame by time
    df = df_orig.sort_values('t')
    if dataset_type == 'caltech101':
        # Convert time to milliseconds
        df['t_ms'] = df['t'] * 1000
        t_w = '5ms'
    else:
        df['t_ms'] = df['t']
        t_w = '10000ms'
    # Erase time offset
    df['t_ms'] = df['t_ms'] - df.loc[0, 't_ms']
    # Set the time column as the index
    df.set_index(pd.to_datetime(df['t_ms'], unit='ms'), inplace=True)
    # Resample the DataFrame at 1 millisecond intervals and calculate the sum of events
    df_resampled = df['t'].astype(bool).resample(t_w).sum()
    # Forward fill missing values (events) and replace NaN with 0
    df_resampled = df_resampled.ffill().fillna(0)
    return df_resampled


# Adapted from https://gist.github.com/foolishflyfox/e30fd8bfbb6a9cee9b1a1fa6144b209c
def plot_sequence_images(image_array: list, **kwargs) -> None:
    ''' Display images sequence as an animation in jupyter notebook

    Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
    '''
    # In case the image array is a list of grayscale images, convert them to RGB keeping the grayscale values
    if len(image_array[0].shape) == 2:
        image_array = [np.stack((img, img, img), axis=-1) for img in image_array] 

    dpi = 72.0
    xpixels, ypixels = image_array[0].shape[:2]
    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    repeat_delay = kwargs.get('repeat_delay', 1)
    flag_repeat = kwargs.get('flag_repeat', True)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array), interval=33, repeat_delay=repeat_delay, repeat=flag_repeat)
    display(HTML(anim.to_html5_video()))


def plot_sequence_images_left_right(left_images: list, right_images: list, **kwargs) -> None:
    ''' Display images sequence as an animation in jupyter notebook

    Args:
        left_images (list of numpy.ndarray): List of left camera images (shape: num_images, height, width, num_channels)
        right_images (list of numpy.ndarray): List of right camera images (shape: num_images, height, width, num_channels)
    '''
    dpi = 72.0
    xpixels, ypixels = left_images[0].shape[:2]
    fig = plt.figure(figsize=(2 * ypixels / dpi, xpixels / dpi), dpi=dpi)

    left_im = plt.figimage(left_images[0])
    right_im = plt.figimage(right_images[0], xo=ypixels)

    def animate(i):
        left_im.set_array(left_images[i])
        right_im.set_array(right_images[i])
        return (left_im, right_im)

    repeat_delay = kwargs.get('repeat_delay', 1)
    flag_repeat = kwargs.get('flag_repeat', True)

    anim = animation.FuncAnimation(fig, animate, frames=len(left_images), interval=33, repeat_delay=repeat_delay, repeat=flag_repeat)
    display(HTML(anim.to_html5_video()))


def jiggle_sequence_images(image_array: list) -> None:
    image_array_backwards = reversed(list(image_array))
    image_array_double = list(image_array) + list(image_array_backwards)
    plot_sequence_images(image_array_double)
