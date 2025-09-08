# Script to compute the prior PDF of the different datasets and tasks

# Main file that contains the class EventSampler and its children
import numpy as np
from pathlib import Path

# To avoid "ModuleNotFoundError: No module named 'external'" error
# append "external" folder to the sys path, which is in the parent directory
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import external.processor as ep
import external.misc as misc
import utils as ut

def main_rvt():
    # Get sequence information from sequence
    test_seq_root_path = dataset_path_test / "17-04-14_14-59-17_854500000_914500000"
    test_seq_evs_path = Path(str(test_seq_root_path) + "_td.dat.h5")
    event_processor = ep.EventProcessorRVT(evs_path=test_seq_evs_path)
    sensor_size = [event_processor.size_y, event_processor.size_x]

    # Read all bboxes
    bbox_paths = [p for p in dataset_path_test.iterdir() if p.suffix == '.npy']
    bbox_all = []

    for bbox_path in bbox_paths:
        bbox = np.load(str(bbox_path))
        bbox_all += list(bbox)

    bbox_all = np.array(bbox_all)

    bounding_boxes = [(b[1], b[2], b[3], b[4]) for b in bbox_all]
    # bounding_boxes = [bounding_boxes[0]]
    heatmap = misc.generate_heatmap_bboxes(bounding_boxes, image_size=sensor_size, sigma=5)
    
    # Heatmap to pdf
    prior_pdf = heatmap / heatmap.sum()

    # Save heatmap
    save_path = dataset_path / Path('priors')  # Can be "sampling/cvpr_rebuttal"
    save_path.mkdir(parents=True, exist_ok=True)

    save_file_path = save_path / "bboxes_test.npy"
    np.save(str(save_file_path), prior_pdf)

    save_plot_path = save_file_path.with_suffix('.pdf')
    misc.save_plot_heatmap_bboxes(prior_pdf, save_name=save_plot_path)


def main():
    pass


if __name__ == "__main__":
    # Paths (to be read from YAML config file)
    DATASET_ROOT_PATH = Path('/home/user/datasets/gen1')

    dataset_path = DATASET_ROOT_PATH / Path('original/events_original')  # Can be "sampling/cvpr_rebuttal"
    dataset_path_train = dataset_path / "train"
    dataset_path_val = dataset_path / "val"
    dataset_path_test = dataset_path / "test"

    # Generate heatmap (prior PDF)
    main_rvt()
    
    # Save heatmap 
    # save_heatmap(heatmap)
