# Script to convert datasets from HSERGB / BSRGB to something that we can use
# Original dataset structure
# .
# ├── close
# │   └── test
# │       ├── baloon_popping
# │       │   ├── events_aligned
# │       │   |   ├── 000000.npz
# │       │   |   ├── 000001.npz
# │       │   |   └── ...
# │       │   └── images_corrected
# │       │       ├── 000000.png
# │       │       ├── 000001.png
# │       │       └── ...
# │       ├── candle
# │       │   ├── events_aligned
# │       │   └── images_corrected
# │       ...
# │
# └── far
#     └── test
#         ├── bridge_lake_01
#         │   ├── events_aligned
#         │   └── images_corrected
#         ├── bridge_lake_03
#         │   ├── events_aligned
#         │   └── images_corrected
#         ...

# Dataset structure that we want
# └── results_folder
#     ├── sequence_0
#     │   ├── GT               <------ this is where ground truth images are stored
#     │   │   ├── 000000.png
#     │   │   ├── 000001.png
#     │   │   └── ...
#     │   ├── method_0          <------ this is the first method to be evaluated
#     │   │   ├── 000000.png
#     │   │   ├── 000001.png
#     │   │   └── ...
#     │   ├── method_1
#     │   │   ├── 000000.png
#     │   │   ├── 000001.png
#     │   │   └── ...
#     │   └── ...
#     ├── sequence_1
#     │   ├── GT
#     │   │   ├── 000000.png
#     │   │   ├── 000001.png
#     │   │   └── ...
#     │   └── ...
#     └── ...


import os
from pathlib import Path
from tqdm import tqdm


class DatasetConverter:
    def __init__(self, orig_path, res_path):
        self.orig_path = orig_path
        self.res_path = res_path
        # Variables
        self.seqs_path_list = []

    def find_seqs(self):
        # Class to be done for every child
        pass

    def convert_dataset(self):
        # Class to be done for every child
        pass


class DatasetConverterHSERGB(DatasetConverter):
    def __init__(self, orig_path, res_path, **kwargs):
        super().__init__(orig_path, res_path)
        # Paths
        self.timelens_path = kwargs.get('timelens_path', res_path)
        # Init variables
        self.find_seqs()

    def find_seqs(self):
        # Find sequence paths and save them to a list
        for root, dirs, files in os.walk(self.orig_path):
            if "images_corrected" in dirs:
                self.seqs_path_list.append(Path(root))

    @staticmethod
    def filter_images_with_events(seq_path_imgs, seq_path_evs, ext='.png'):
        # Filter images based on the events we have available
        new_img_evs_tuple_paths = []
        new_evs_path_list = []  # Not sure if we have to discard the 1st event
        for evs_path in seq_path_evs.iterdir():
            img_path = (seq_path_imgs / evs_path.name).with_suffix(ext)
            if img_path.is_file():
                new_img_evs_tuple_paths.append((img_path, evs_path))

        return new_img_evs_tuple_paths

    @staticmethod
    def convert_dataset_for_timelens(seq_path_res, img_evs_tuples):
        # Convert to timelens type dataset with new data so we can easily do tests on it
        # Make images and events folders and create soft links to the corresponding directory in the results folder
        seq_path_res_img = seq_path_res / 'images'
        seq_path_res_evs = seq_path_res / 'events' / 'original'
        seq_path_res_img.mkdir(exist_ok=True)
        seq_path_res_evs.mkdir(parents=True, exist_ok=True)

        # Timestamp file to symlink
        timestamp_path = img_evs_tuples[0][0].parent / 'timestamp.txt'
        res_timestamp_path = seq_path_res_img / timestamp_path.name
        tstamp_rel = os.path.relpath(timestamp_path, res_timestamp_path.parent)
        if not res_timestamp_path.is_file():
            res_timestamp_path.symlink_to(tstamp_rel)

        # Symlinks for images and events
        for img, evs in img_evs_tuples:
            res_img_path = seq_path_res_img / img.name
            res_evs_path = seq_path_res_evs / evs.name
            img_rel = os.path.relpath(img, res_img_path.parent)
            evs_rel = os.path.relpath(evs, res_evs_path.parent)
            if not res_img_path.is_file():
                res_img_path.symlink_to(img_rel)
                res_evs_path.symlink_to(evs_rel)

    @staticmethod
    def convert_dataset_for_eval(seq_path_res, img_evs_tuples):
        # Convert to eval type dataset with new data so we can easily do tests on it
        # Make GT folder and create soft links to the images in the corresponding directory in the results folder
        seq_path_res_gt = seq_path_res / 'GT'
        seq_path_res_gt.mkdir(exist_ok=True)

        for img, _ in img_evs_tuples:
            res_img_path = seq_path_res_gt / img.name
            img_rel = os.path.relpath(img, res_img_path.parent)
            if not res_img_path.is_file():
                res_img_path.symlink_to(img_rel)

    def convert_dataset(self):
        self.res_path.mkdir(parents=True, exist_ok=True)
        self.timelens_path.mkdir(parents=True, exist_ok=True)  # If None is self.res_path

        for seq_path in tqdm(self.seqs_path_list):
            seq_path_imgs = seq_path / "images_corrected"
            seq_path_evs = seq_path / "events_aligned"
            # Filter images based on the events we have available
            img_evs_tuples = self.filter_images_with_events(seq_path_imgs, seq_path_evs)
            # Make path for every sequence in the converted path
            seq_path_res_eval = self.res_path / seq_path.name
            seq_path_res_eval.mkdir(exist_ok=True)
            seq_path_res_timelens = self.timelens_path / seq_path.name
            seq_path_res_timelens.mkdir(exist_ok=True)

            self.convert_dataset_for_eval(seq_path_res_eval, img_evs_tuples)
            self.convert_dataset_for_timelens(seq_path_res_timelens, img_evs_tuples)


def main(orig_path, res_path, timelens_path=None):
    dataset_converter = DatasetConverterHSERGB(orig_path, res_path, timelens_path=timelens_path)
    dataset_converter.convert_dataset()


if __name__ == "__main__":
    # Set the paths to the original dataset and the results folder
    original_dataset_path = Path('/home/user/datasets/hs-ergb/hsergb')  # Host
    results_folder_path = Path('/home/user/datasets/hs-ergb/eval_interpolation/hsergb')
    timelens_folder_path = Path('/home/user/datasets/hs-ergb/sampling/hsergb_conversion/seqs')
    
    main(original_dataset_path, results_folder_path, timelens_folder_path)
