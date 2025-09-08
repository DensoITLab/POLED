# Miscellaneous functions

# Common imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2


# Functions
def df_filter_events_time(df_evs: pd.DataFrame, t_start=0, t_end=1e8) -> pd.DataFrame:
    return df_evs[(df_evs['t'] >= t_start) & (df_evs['t'] <= t_end)]


# Color palette to choose from when calling it (e.g. COLORS_GENERATOR(0) is red, COLORS_GENERATOR(1) is blue, ...) without using matplotlib
def COLORS_GENERATOR(n: int) -> np.array:
    colors = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [0.0, 0.0, 0.0],  # Black
        [1.0, 1.0, 1.0],  # White
        [0.5, 0.5, 0.5],  # Gray
        [0.5, 0.0, 0.0],  # Dark Red
        [0.0, 0.5, 0.0],  # Dark Green
        [0.0, 0.0, 0.5],  # Dark Blue
        [0.5, 0.5, 0.0],  # Dark Yellow
        [0.5, 0.0, 0.5],  # Dark Magenta
        [0.0, 0.5, 0.5],  # Dark Cyan
        [0.5, 0.5, 0.5],  # Dark Gray
        [0.5, 0.5, 0.5],  # Dark White
        [0.25, 0.25, 0.25],  # Dark Gray
        [0.75, 0.75, 0.75],  # Light Gray
        [0.75, 0.0, 0.0],  # Light Red
        [0.0, 0.75, 0.0],  # Light Green
        [0.0, 0.0, 0.75],  # Light Blue
        [0.75, 0.75, 0.0],  # Light Yellow
        [0.75, 0.0, 0.75],
        [0.0, 0.75, 0.75],
        [0.75, 0.75, 0.75],
        [0.25, 0.25, 0.25]])

    if n >= len(colors):
        return colors[int(n % len(colors))]
    else:
        return colors[int(n)]


def add_frame_num(frame: np.array, frame_num: int) -> np.array:
    f = frame.copy()
    cv2.putText(f, str(frame_num), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return f


def add_frame_num_list(frames: list) -> list:
    frames_out = []
    for i, f in enumerate(frames):
        f_out = add_frame_num(f, i)
        frames_out.append(f_out)


def df2eventsimg(df_evs: pd.DataFrame, **kwargs) -> list:
    t_current = kwargs.get('t_start', 0)
    t_window = kwargs.get('t_window', 10e-3)  # 10 ms, as dataset is in seconds
    t_max = df_evs['t'].to_numpy()[-1]

    # Case where the events have a cluster label
    is_color = kwargs.get('is_color', False)

    # Sensor info
    sensor_y = kwargs.get('sensor_y', int(df_evs['y'].max()) + 1)
    sensor_x = kwargs.get('sensor_x', int(df_evs['x'].max()) + 1)

    # Output frame info
    frame_height = kwargs.get('frame_height', 180)
    frame_width = kwargs.get('frame_width', 240)

    frame_idx = 0
    frames = []

    while t_current < t_max:
        # Mask events from t0 -> twindow
        mask = (df_evs['t'] > t_current) & (df_evs['t'] <= t_current + t_window)
        # Generate event frame
        x_ = df_evs['x'][mask]
        y_ = df_evs['y'][mask]
        t_ = df_evs['t'][mask]
        if is_color:
            ev_color_ = df_evs['id'][mask]
        else:
            ev_color_ = df_evs['p'][mask]

        evs_np = np.vstack([y_, x_, t_, ev_color_]).T
        image = events2img(evs_np, (sensor_y, sensor_x), is_color=is_color)
        image = image.resize((frame_width, frame_height))
        image_np = np.array(image)
        image_np = add_frame_num(image_np, frame_idx)
        frames.append(image_np)

        frame_idx += 1
        t_current += t_window

    return frames


def events2img(events: np.ndarray, image_shape: tuple, **kwargs) -> Image.Image:
    """Visualize events with polarity color.

    Args:
        events (np.ndarray): _description_
        image_shape (tuple): _description_
        kwargs: _description_

    Returns:
        _type_: _description_
    """
    is_color = kwargs.get('is_color', False)

    background_color = 255
    image = (
            np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8)
            * background_color
    )  # RGBA channel

    if len(events) > 0:
        events[:, 0] = np.clip(events[:, 0], 0, image_shape[0] - 1)
        events[:, 1] = np.clip(events[:, 1], 0, image_shape[1] - 1)
        if is_color:
            # Case where the events have a cluster label
            # Check events type
            colors = np.array([COLORS_GENERATOR(e[3]) for e in events]) * 255
        else:
            colors = np.array([(255, 0, 0) if e[3] == 1 else (0, 0, 255) for e in events])

        image[events[:, 0].astype(np.int32), events[:, 1].astype(np.int32), :] = colors

    image = Image.fromarray(image)
    return image


def generate_tracklet_animation(frames_in:list, df_tracks:pd.DataFrame, **kwargs) -> list:
    # Combine the frames and the tracklets in a fancy animation

    t_current = 0
    t_window = kwargs.get('t_window', 10e-3)  # 10 ms, as dataset is in seconds
    t_keep = kwargs.get('t_keep', 0.5)  # 500 ms

    # Create a list of images
    frames_out = []

    for i, f_in in enumerate(frames_in):
        # Get the tracklets for the current frame
        #mask = (df_tracks['t'] > t_current) & (df_tracks['t'] <= t_current + t_window)
        mask = (df_tracks['t'] > t_current - t_keep) & (df_tracks['t'] <= t_current + t_window)
        df_ = df_tracks[mask]

        # Draw the tracklets
        f_out = draw_tracklets(f_in, df_)
        frames_out.append(f_out)

        t_current += t_window

    return frames_out


def draw_tracklets(frame:np.array, df_tracks:pd.DataFrame) -> np.array:
    # Tracklets are in the format: x, y, t, id

    f_out = frame.copy()

    # Draw the tracklets
    for i, tracklet in df_tracks.iterrows():
        x = int(tracklet['x'])
        y = int(tracklet['y'])
        id = tracklet['id']

        # Draw the tracklet
        cv2.circle(f_out, (x, y), 2, COLORS_GENERATOR(id) * 255, -1)

    return f_out


def generate_event_frame_diff(df_evs: pd.DataFrame, **kwargs) -> np.array:
    sensor_y = kwargs.get('sensor_y', int(df_evs['y'].max()) + 1)
    sensor_x = kwargs.get('sensor_x', int(df_evs['x'].max()) + 1)

    # Convert polarities from [0, 1] to [-1, 1]
    df_evs['p'] = df_evs['p'].replace({0: -1})

    # Compute image by doing the summation of the polarities per pixel
    image_array = np.zeros((sensor_y, sensor_x))  # We can have negative values
    for i, row in df_evs.iterrows():
        image_array[int(row['y']), int(row['x'])] += row['p']

    return image_array

#+ HEATMAPS +#

def generate_heatmap_bboxes(bounding_boxes, image_size=(100, 100), sigma=1):
    heatmap = np.zeros(image_size)

    for bbox in bounding_boxes:
        x1, y1, w, h = bbox
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image_size[1], int(x1 + w))
        y2 = min(image_size[0], int(y1 + h))

        for y in range(y1, y2):
            for x in range(x1, x2):
                heatmap[y, x] += 1

    # heatmap = np.exp(-0.5 * ((heatmap / len(bounding_boxes)) / sigma) ** 2)
    heatmap = heatmap / heatmap.max()
    return heatmap


def generate_heatmap_events(df_evs, sensor_size=None):
    # Define bins
    if sensor_size is None:
        ybins = np.arange(0, df_evs['y'].max() + 2)
        xbins = np.arange(0, df_evs['x'].max() + 2)
    else:
        ybins = np.arange(0, sensor_size[0] + 1)
        xbins = np.arange(0, sensor_size[1] + 1)

    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(df_evs['y'], df_evs['x'], bins=(ybins, xbins))
    heatmap += 0.1  # For log

    return heatmap


def plot_heatmap_bboxes(heatmap):
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


def plot_heatmap_events(heatmap):
    # plt.imshow(heatmap, origin='upper', cmap='hot', interpolation='nearest', norm=LogNorm(vmin=heatmap.min()+1, vmax=heatmap.max()))
    plt.imshow(heatmap, cmap='hot', norm=LogNorm(vmin=heatmap.min() + 1, vmax=heatmap.max()))
    # plt.imshow(heatmap, origin='upper', cmap='hot')
    plt.colorbar()
    plt.show()


def save_plot_heatmap_bboxes(heatmap, save_name="test_heatmap.pdf"):
    fig = plt.figure()
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')

    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False
    )

    # plt.colorbar()
    plt.savefig(save_name, format="pdf", bbox_inches="tight")
    # plt.show()


def save_plot_heatmap_events(heatmap, save_name="test_heatmap.pdf"):
    fig = plt.figure()
    # plt.imshow(heatmap, origin='upper', cmap='hot', interpolation='nearest', norm=LogNorm(vmin=heatmap.min()+1, vmax=heatmap.max()))
    plt.imshow(heatmap, cmap='hot', norm=LogNorm(vmin=heatmap.min() + 1, vmax=heatmap.max()))
    # plt.imshow(heatmap, origin='upper', cmap='hot')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False
    )

    # plt.colorbar()
    plt.savefig(save_name, format="pdf", bbox_inches="tight")
    # plt.show()

#- HEATMAPS -#


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
    
    
def save_evs_tdiff(df_evs, save_name, xstart=0, xend=1e8):
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

    plt.savefig(save_name, format="pdf", bbox_inches="tight")


def save_evs_frames(event_frames, save_dir, prefix="evs_frame"):
    for i, frame in tqdm(enumerate(event_frames)):
        save_name = f"{save_dir}/{prefix}_{str(i)}.png"
        cv2.imwrite(save_name, frame)
    
    
def compute_evs_tdiff(df_evs):
    df_evs['delta_t'] = df_evs.groupby(['x', 'y'])['t'].diff()
    return df_evs


def filter_evs_tdiff(df_evs, xstart=0, xend=1e8):
    return df_evs[(df_evs['delta_t'] >= xstart) & (df_evs['delta_t'] <= xend)]


def reset_plot():
    plt.clf()
