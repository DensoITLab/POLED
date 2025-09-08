# Following information's theory, compute the information loss using normalized mutual information.
# Also compute the KL divergence between original and downsampled events.

import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import entropy
from scipy.special import kl_div


class EventMetrics:
    def __init__(self, size_y, size_x, coarse_factor=2):
        self.size_y = size_y
        self.size_x = size_x
        self.coarse_factor = coarse_factor
        self.coarse_size_y = (size_y // coarse_factor) + 1
        self.coarse_size_x = (size_x // coarse_factor) + 1

    def compute_normalized_mutual_information(self, original_evs, downsampled_evs):
        """
        Compute the normalized mutual information between original and downsampled events using sklearn,
        based on a coarser PDF.
        """
        original_coarse, downsampled_coarse = self.events_to_coarse_images(original_evs, downsampled_evs)
        
        # Flatten the coarse event images to 1D arrays
        original_flat = original_coarse.flatten()
        downsampled_flat = downsampled_coarse.flatten()
        
        # Compute normalized mutual information using sklearn
        normalized_mutual_info = normalized_mutual_info_score(original_flat, downsampled_flat)
        return normalized_mutual_info

    def compute_kl_divergence(self, original_evs, downsampled_evs):
        """
        Compute the KL divergence between original and downsampled events using SciPy,
        based on a coarser PDF.
        """
        original_hist = self.events_to_coarse_histogram(original_evs)
        downsampled_hist = self.events_to_coarse_histogram(downsampled_evs)
        
        original_hist = self.normalize_histogram(original_hist)
        downsampled_hist = self.normalize_histogram(downsampled_hist)
        
        # Use SciPy's kl_div function
        kl = np.sum(kl_div(original_hist.flatten() + 1e-10, downsampled_hist.flatten() + 1e-10))
        return kl

    def events_to_coarse_images(self, original_evs, downsampled_evs):
        original_coarse = np.zeros((self.coarse_size_y, self.coarse_size_x), dtype=np.int32)
        downsampled_coarse = np.zeros((self.coarse_size_y, self.coarse_size_x), dtype=np.int32)
        
        np.add.at(original_coarse, (original_evs['y'] // self.coarse_factor, original_evs['x'] // self.coarse_factor), 1)
        np.add.at(downsampled_coarse, (downsampled_evs['y'] // self.coarse_factor, downsampled_evs['x'] // self.coarse_factor), 1)
        
        return original_coarse, downsampled_coarse

    def events_to_coarse_histogram(self, events):
        coarse_y = events['y'] // self.coarse_factor
        coarse_x = events['x'] // self.coarse_factor
        hist, _ = np.histogramdd(np.column_stack((coarse_y, coarse_x)), bins=(self.coarse_size_y, self.coarse_size_x))
        return hist

    def normalize_histogram(self, hist):
        return hist / np.sum(hist)

def main(event_processor, downsampled_event_processor, name):
    total_nmi = 0
    total_kl = 0
    num_packets = 0
    metrics = EventMetrics(event_processor.size_y + 1, event_processor.size_x + 1)

    while True:
        # Read events
        original_evs = event_processor.read_next_event_packets(num_packets=1)
        downsampled_evs = downsampled_event_processor.read_next_event_packets(num_packets=1)

        if original_evs == -1 or downsampled_evs == -1:
            break

        if original_evs['t'].size == 0 or downsampled_evs['t'].size == 0:
            continue

        nmi = metrics.compute_normalized_mutual_information(original_evs, downsampled_evs)
        kl = metrics.compute_kl_divergence(original_evs, downsampled_evs)
        
        total_nmi += nmi
        total_kl += kl
        num_packets += 1
   
    return total_nmi / num_packets, total_kl / num_packets


if __name__ == "__main__":
    from pathlib import Path
    
    # To avoid "ModuleNotFoundError: No module named 'external'" error
    # append "external" folder to the sys path, which is in the parent directory
    import sys
    external_path = Path(__file__).resolve().parent.parent / "external"
    print(external_path)
    sys.path.append(str(external_path.parent))

    # Do an example run using the EventProcessor
    from external.processor import EventProcessorCaltech101RPG
    import numpy as np
    import matplotlib.pyplot as plt

    EV_CAMS_DATASET_ROOT = Path("/home/user/datasets")
    #exp_name = "cvpr25"
    exp_name = "cvpr25-sigmoid"
    DEBUG = True

    # Variables
    sampling_rate = "0.1"
    t_window = 10 # 5ms 

    if DEBUG:
        seq = "Faces_easy/Faces_easy_0.npy"
        EV_PATH = EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/original/testing/{seq}"
        EV_PATH_KL_DEBUG = EV_PATH
        EV_PATH_SAMPLED_UNIFORM = EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/sampling/{exp_name}/events_uniform_{sampling_rate}-1/testing/{seq}"
        EV_PATH_SAMPLED = EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/sampling/{exp_name}/events_accum_{sampling_rate}-1/testing/{seq}"
        EV_PATH_SAMPLED_POI = EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/sampling/{exp_name}/events_poisson_{sampling_rate}-1/testing/{seq}"
        EV_PATH_SAMPLED_NN = EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/sampling/{exp_name}/events_pointnet_{sampling_rate}-1/testing/{seq}"
        #EV_PATH_SAMPLED_PF = f"tmp/faces_easy/Faces_easy_0.npy"       

        # Initialize the event processor
        event_processor = EventProcessorCaltech101RPG(evs_file=EV_PATH, bin_size=t_window)
        event_processor_kl = EventProcessorCaltech101RPG(evs_file=EV_PATH_KL_DEBUG, bin_size=t_window)
        event_processor_uniform = EventProcessorCaltech101RPG(evs_file=EV_PATH_SAMPLED_UNIFORM, bin_size=t_window)
        event_processor_sampled = EventProcessorCaltech101RPG(evs_file=EV_PATH_SAMPLED, bin_size=t_window)
        event_processor_poi = EventProcessorCaltech101RPG(evs_file=EV_PATH_SAMPLED_POI, bin_size=t_window)
        event_processor_nn = EventProcessorCaltech101RPG(evs_file=EV_PATH_SAMPLED_NN, bin_size=t_window)

        # Run the metrics
        nmi_kl, kl_kl = main(event_processor, event_processor_kl, "Original")
        event_processor.reset()
        nmi_uniform, kl_uniform = main(event_processor, event_processor_uniform, "Uniform")
        event_processor.reset()
        nmi_accum, kl_accum = main(event_processor, event_processor_sampled, "Accum")
        event_processor.reset()
        nmi_poi, kl_poi = main(event_processor, event_processor_poi, "Poisson")
        event_processor.reset()
        nmi_nn, kl_nn = main(event_processor, event_processor_nn, "PointNet")

        # Print or store the metrics
        print(f"Original - NMI: {nmi_kl:.4f}, KL Divergence: {kl_kl:.4f}")
        print(f"Uniform Sampling - NMI: {nmi_uniform:.4f}, KL Divergence: {kl_uniform:.4f}")
        print(f"Accumulator Sampling - NMI: {nmi_accum:.4f}, KL Divergence: {kl_accum:.4f}")
        print(f"Poisson Sampling - NMI: {nmi_poi:.4f}, KL Divergence: {kl_poi:.4f}")
        print(f"PointNet Sampling - NMI: {nmi_nn:.4f}, KL Divergence: {kl_nn:.4f}")
    
    else:
        sampling_rates = ["0.1", "0.5", "0.9"]
        for sampling_rate in sampling_rates:
            EV_PATH = EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/original/testing"
            EV_PATH_SAMPLED_UNIFORM = EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/sampling/{exp_name}/events_uniform_{sampling_rate}-1/testing"
            EV_PATH_SAMPLED = EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/sampling/{exp_name}/events_accum_{sampling_rate}-1/testing"
            EV_PATH_SAMPLED_POI = EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/sampling/{exp_name}/events_poisson_{sampling_rate}-1/testing"
            EV_PATH_SAMPLED_NN = EV_CAMS_DATASET_ROOT / f"N-Caltech101/rpg_event_representation_learning/sampling/{exp_name}/events_pointnet_{sampling_rate}-1/testing"

            # Traverse the testing folder to get the different categories and the sequences in each category, run the metrics for each sequence
            total_nmi = {"Uniform": 0, "Accum": 0, "Poisson": 0, "PointNet": 0}
            total_kl = {"Uniform": 0, "Accum": 0, "Poisson": 0, "PointNet": 0}
            category_metrics = {}
            num_packets = 0

            for category_path in EV_PATH.iterdir():
                if category_path.is_dir():

                    print(f"Processing category: {category_path.name}")
                    category_metrics[category_path.name] = {
                        "nmi": {"Uniform": 0, "Accum": 0, "Poisson": 0, "PointNet": 0},
                        "kl": {"Uniform": 0, "Accum": 0, "Poisson": 0, "PointNet": 0},
                        "count": 0
                    }

                    for seq_path in category_path.iterdir():
                        if seq_path.is_file() and seq_path.suffix == '.npy':
                            print(f"Processing sequence: {seq_path.name}")
                            
                            # Construct paths for sampled sequences
                            uniform_path = EV_PATH_SAMPLED_UNIFORM / category_path.name / seq_path.name
                            sampled_path = EV_PATH_SAMPLED / category_path.name / seq_path.name
                            poi_path = EV_PATH_SAMPLED_POI / category_path.name / seq_path.name
                            nn_path = EV_PATH_SAMPLED_NN / category_path.name / seq_path.name
                            
                            if uniform_path.exists() and sampled_path.exists():
                                # Initialize the event processors
                                event_processor = EventProcessorCaltech101RPG(evs_file=seq_path, bin_size=t_window)
                                event_processor_uniform = EventProcessorCaltech101RPG(evs_file=uniform_path, bin_size=t_window)
                                event_processor_sampled = EventProcessorCaltech101RPG(evs_file=sampled_path, bin_size=t_window)
                                event_processor_poi = EventProcessorCaltech101RPG(evs_file=poi_path, bin_size=t_window)
                                event_processor_nn = EventProcessorCaltech101RPG(evs_file=nn_path, bin_size=t_window)

                                nmi_uniform, kl_uniform = main(event_processor, event_processor_uniform, "Uniform")
                                event_processor.reset()
                                nmi_accum, kl_accum = main(event_processor, event_processor_sampled, "Accum")
                                event_processor.reset()
                                nmi_poi, kl_poi = main(event_processor, event_processor_poi, "Poisson")
                                event_processor.reset()
                                nmi_nn, kl_nn = main(event_processor, event_processor_nn, "PointNet")

                                # Update global metrics
                                total_nmi["Uniform"] += nmi_uniform
                                total_kl["Uniform"] += kl_uniform
                                total_nmi["Accum"] += nmi_accum
                                total_kl["Accum"] += kl_accum
                                total_nmi["Poisson"] += nmi_poi
                                total_kl["Poisson"] += kl_poi
                                total_nmi["PointNet"] += nmi_nn
                                total_kl["PointNet"] += kl_nn

                                # Update category metrics
                                category_metrics[category_path.name]["nmi"]["Uniform"] += nmi_uniform
                                category_metrics[category_path.name]["kl"]["Uniform"] += kl_uniform
                                category_metrics[category_path.name]["nmi"]["Accum"] += nmi_accum
                                category_metrics[category_path.name]["kl"]["Accum"] += kl_accum
                                category_metrics[category_path.name]["nmi"]["Poisson"] += nmi_poi
                                category_metrics[category_path.name]["kl"]["Poisson"] += kl_poi
                                category_metrics[category_path.name]["nmi"]["PointNet"] += nmi_nn
                                category_metrics[category_path.name]["kl"]["PointNet"] += kl_nn
                                category_metrics[category_path.name]["count"] += 1

                                num_packets += 1
                            else:
                                print(f"Skipping {seq_path.name}: Not all sampled sequences exist")

            # Print nice table for per category metrics
            print("\nPer Category Metrics:")
            for category, metrics in category_metrics.items():
                print(f"\n{category}:")
                print(f"{'Method':<10} {'NMI':<10} {'KL':<10}")
                print("-" * 30)
                for method in ["Uniform", "Accum", "Poisson", "PointNet"]:
                    avg_nmi = metrics["nmi"][method] / metrics["count"]
                    avg_kl = metrics["kl"][method] / metrics["count"]
                    print(f"{method:<10} {avg_nmi:<10.4f} {avg_kl:<10.4f}")

            # Print nice table for global metrics
            print("\nGlobal Metrics:")
            print(f"{'Method':<10} {'NMI':<10} {'KL':<10}")
            print("-" * 30)
            for method in ["Uniform", "Accum", "Poisson", "PointNet"]:
                print(f"{method:<10} {total_nmi[method] / num_packets:<10.4f} {total_kl[method] / num_packets:<10.4f}")


            # Save metrics to a file
            with open(f"results/{exp_name}-information_loss_metrics_{sampling_rate}.txt", "w") as f:
                f.write(f"Global Metrics:\n")
                f.write(f"{'Method':<10} {'NMI':<10} {'KL':<10}\n")
                f.write(f"-" * 30 + "\n")
                for method in ["Uniform", "Accum", "Poisson", "PointNet"]:
                    f.write(f"{method:<10} {total_nmi[method] / num_packets:<10.4f} {total_kl[method] / num_packets:<10.4f}\n")

                f.write(f"\nPer Category Metrics:\n")
                
                for category, metrics in category_metrics.items():
                    f.write(f"\n{category}:\n")
                    f.write(f"{'Method':<10} {'NMI':<10} {'KL':<10}\n")
                    f.write(f"-" * 30 + "\n")
                    for method in ["Uniform", "Accum", "Poisson", "PointNet"]:
                        avg_nmi = metrics["nmi"][method] / metrics["count"]
                        avg_kl = metrics["kl"][method] / metrics["count"]
                        f.write(f"{method:<10} {avg_nmi:<10.4f} {avg_kl:<10.4f}\n")
