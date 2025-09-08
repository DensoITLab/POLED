import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Event sampling arguments')

    # Configuration file
    parser.add_argument('--cfg', type=str, default='/home/user/app/config/ev_sampling/iccv25.yaml',
                        help='Path to the experiment configuration file')

    # Paths
    parser.add_argument("--seqs_path", default="", help="Path to sequences")
    parser.add_argument("--ev_path", default="/home/agirbau/work/event_cameras/rpg_timelens/example/example3/events",
                        help="Path to events")
    parser.add_argument("--ev_save_path", default="/home/agirbau/work/event_cameras/rpg_timelens/example/example3",
                        help="Path to save processed events")
    parser.add_argument("--img_path", default="/home/agirbau/work/event_cameras/rpg_timelens/example/example3/images",
                        help="Path to images")
    parser.add_argument("--tstmp_path", default="/home/agirbau/work/event_cameras/rpg_timelens/example/example3/images",
                        help="Path to timestamps")

    # Variables
    parser.add_argument('--dataset_name', type=str, choices=['NCaltech101', 'hsergb', 'gen1', 'esfp'],
                        help='Type of dataset to be sampled')
    parser.add_argument('--dataset_type', type=str, choices=['caltech101', 'cars', 'timelens', 'rvt', 'ramnet', 'esfp'],
                        help='Type of dataset to be sampled')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--sampler', type=str, choices=['det', 'detTemp', 'detSpat', 'uniform', 'accum', 'accum_hier',
                                                        'weighted', 'weightedSepPols', 'KDE',
                                                        'poisson', 'pointnet', 'evDownNavi'],
                        help='Sampling strategy')
    parser.add_argument('--exp_name', type=str, help='Experiment name')
    parser.add_argument('--num_exp', type=int, help='Experiment number')
    parser.add_argument('--prob_init', type=str, help='Initial probability for sampling (0.prob_accept)')

    # Hyperparameters
    parser.add_argument('--t_window_size', type=int, help='Temporal window size (ms)')
    parser.add_argument('--coarse_size', type=int, help='Coarse size for accumulation')
    parser.add_argument('--particle_size', type=int, help='Particle base size (AxA)')
    parser.add_argument('--n_particles', type=int, help='Number of particles')
    parser.add_argument('--alpha', type=float, help='Alpha factor for distribution fusion')
    parser.add_argument('--alpha_prior', type=str, help='Prior temperature factor for RVT')
    parser.add_argument('--l_accept', type=float, help='Lambda for accepted events')
    parser.add_argument('--l_reject', type=float, help='Lambda for rejected events')

    # Deprecated
    parser.add_argument('--ker_size', type=int, help='Spatial kernel size for random sampling')

    # Code configuration
    parser.add_argument('--num_proc', type=int, help='Number of processes')

    # Flags
    parser.add_argument('--t_surfaces', action='store_true', help='Enable decreasing exponential distribution')
    parser.add_argument('--d', action='store_true', help='Enable debugging mode')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    sysargs = parser.parse_args()
