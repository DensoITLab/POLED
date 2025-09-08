# Script to prepare NCaltech101 for sampling
# Original dataset structure

# NCaltech101
# ├── testing
# │    ├── BACKGROUND_Google
# │    │   ├── BACKGROUND_Google_0.npy
# |    │   ├── BACKGROUND_Google_1.npy
# │    │   └── ...
# │    ├── Faces_easy
# │    │   ├── Faces_easy_0.npy
# │    │   └── Faces_easy_1.npy
# │    │   └── ...
# │    ...
# │
# └── training
# │   ├── BACKGROUND_Google
# │   │   ├── BACKGROUND_Google_0.npy
# │   │   ├── BACKGROUND_Google_1.npy
# │   │   └── ...
# │   ├── Faces_easy
# │   │   ├── Faces_easy_0.npy
# │   │   └── Faces_easy_1.npy
# │   └── ...
# │
# │
# └── validation
#     ├── BACKGROUND_Google
#     │   ├── BACKGROUND_Google_0.npy
#     │   ├── BACKGROUND_Google_1.npy
#     │   └── ...
#     ├── Faces_easy
#     │   ├── Faces_easy_0.npy
#     │   └── Faces_easy_1.npy
#     └── ...


# Dataset structure that we want
# Example: N-Caltech101/rpg_event_representation_learning/N-Caltech101/original/testing/Faces_easy/Faces_easy_0.npy
# EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/N-Caltech101/sampling/uniform/info/testing/Faces_easy/Faces_easy_0.npy"

# N-Caltech101
# └── rpg_event_representation_learning
#     └── N-Caltech101
#         ├── original
#         │   ├── testing
#         │   │   ├── Faces_easy
#         │   │   │   ├── Faces_easy_0.npy
#         │   │   │   ├── Faces_easy_1.npy
#         │   │   │   └── ...
#         │   ├── training
#         │   │   ├── Faces_easy
#         │   │   │   ├── Faces_easy_0.npy
#         │   │   │   └── Faces_easy_1.npy
#         │   │   └── ...
#         │   └── validation
#         │       ├── Faces_easy
#         │       │   ├── Faces_easy_0.npy
#         │       │   ├── Faces_easy_1.npy
#         │       │   └── ...
#         └── sampling
#             ├── uniform
#             │   ├── info  <-- contains the information about the sampling method, e.g. sampling rate or bitrate
#             │   │   ├── testing
#             │   │   │   ├── Faces_easy
#             │   │   │   │   ├── Faces_easy_0.npy
#             │   │   │   │   ├── Faces_easy_1.npy
#             │   │   │   │   └── ...
#             │   │   ├── training
#             │   │   │   ├── Faces_easy
#             │   │   │   │   ├── Faces_easy_0.npy
#             │   │   │   │   └── Faces_easy_1.npy
#             │   │   │   └── ...
#             │   └── validation
#             │       ├── Faces_easy
#             │       │   ├── Faces_easy_0.npy
#             │       │   ├── Faces_easy_1.npy
#             │       │   └── ...
#             └── histogram
#                 ├── info
#                 │   ├── testing
#                 │   │   ├── Faces_easy
#                 │   │   │   ├── Faces_easy_0.npy
#                 │   │   │   ├── Faces_easy_1.npy
#                 │   │   │   └── ...
#                 │   ├── training
#                 │   │   ├── Faces_easy
#                 │   │   │   ├── Faces_easy_0.npy
#                 │   │   │   └── Faces_easy_1.npy
#                 │   │   └── ...
#                 └── validation
#                     ├── Faces_easy
#                     │   ├── Faces_easy_0.npy
#                     │   ├── Faces_easy_1.npy
#                     │   └── ...
#                     └── ...
#
# The script will create the necessary directories and copy the files to the correct location


import os
from pathlib import Path

class DatasetConverterCaltech101:
    def __init__(self, orig_path, res_path):
        self.orig_path = Path(orig_path)
        self.res_path = Path(res_path)

    def convert_dataset(self):
        self.res_path.mkdir(parents=True, exist_ok=True)
        os.chmod(self.res_path, 0o777)
        
        res_original_path = self.res_path / 'original'
        res_original_path.mkdir(exist_ok=True)
        os.chmod(res_original_path, 0o777)
        
        res_sampling_path = self.res_path / 'sampling'
        res_sampling_path.mkdir(exist_ok=True)
        os.chmod(res_sampling_path, 0o777)

        for split in ['testing', 'training', 'validation']:
            src_path = self.orig_path / split
            dst_path = res_original_path / split

            if dst_path.is_symlink():
                os.unlink(dst_path)
            
            print(f"Creating symlink from {src_path} to {dst_path}")
            os.symlink(src_path, dst_path)
            

def main(orig_path, res_path):
    dataset_converter = DatasetConverterCaltech101(orig_path, res_path)
    dataset_converter.convert_dataset()


if __name__ == "__main__":
    ORIG_PATH = Path("/petaco/andreu2/datasets/N-Caltech101/N-Caltech101")
    RES_PATH = Path("/petaco/andreu2/datasets/N-Caltech101/rpg_event_representation_learning")

    main(ORIG_PATH, RES_PATH)