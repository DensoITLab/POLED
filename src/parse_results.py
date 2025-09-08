# Copyright: Andreu Girbau-Xalabarder, 2025
# We provide the code as-is,
# without warranty of any kind, express or implied.

# Script to parse the results of the experiments and generate the tables and figures
import argparse
from pathlib import Path
import pandas as pd
import re
import shlex
from tabulate import tabulate
import yaml

import utils as ut


class ResultsParser:
    def __init__(self):
        self.list_df_datasets_logs = []
        self.logs_dict = None
        # Vars
        self.common_metrics = ['Size', 'Path', 'infer_time']

    def parse_logs(self, logs_dict):
        self.list_df_datasets_logs = []
        self.logs_dict = logs_dict

        # Results
        df = self.parse_results(logs_dict['method'])
        if df.empty:
            return -1

        self.list_df_datasets_logs.append(df)
        # Size
        df = self.parse_size(logs_dict['log_size'])
        self.list_df_datasets_logs.append(df)
        # Time
        df = self.parse_time(logs_dict['log_time'])
        self.list_df_datasets_logs.append(df)

        # Specific logs
        if self.logs_dict['log_sampling_time'].exists():
            df = self.parse_time(logs_dict['log_sampling_time'], time_type='sampling_time')
            self.list_df_datasets_logs.append(df)
        if self.logs_dict['log_preproc_time'].exists():
            df = self.parse_time(logs_dict['log_preproc_time'], time_type='preproc_time')
            self.list_df_datasets_logs.append(df)

        # Merge the dataframes
        df_res = self.merge_logs(logs_dict['method'])
        
        return df_res

    def parse_results(self, method):
        if method == 'rpg_replearning':
            df_res = self.parse_res_rpg_replearning(self.logs_dict['log_results'])
        elif method == 'rvt':
            df_res = self.parse_res_rvt(self.logs_dict['log_results'])
        elif method == 'timelens':
            df_res = self.parse_res_timelens(self.logs_dict['log_results'])
        elif method == 'esfp':
            df_res = self.parse_res_esfp(self.logs_dict['log_results'])
        else:
            raise ValueError(f"Method {method} not recognized")

        return df_res

    def merge_logs(self, method):
        df_res = pd.DataFrame()

        if method == 'rpg_replearning':
            df_res = self.merge_logs_rpg_replearning()
        if method == 'rvt':
            df_res = self.merge_logs_rvt()
        if method == 'timelens':
            df_res = self.merge_logs_timelens()
        if method == 'esfp':
            df_res = self.merge_logs_esfp()

        return df_res

    @staticmethod
    def parse_res_rpg_replearning(file_path):
        # Initialize storage for extracted data
        data = {
            'Checkpoint': '',
            'Test Dataset': '',
            'Batch Size': 1,
            'Device': 0,
            'Test Loss': -1,
            'Test Accuracy': -1,
            'Samples/Sec': 0
        }

        df_dummy = pd.DataFrame([data])

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()

                # Extract key-value pairs from the structured lines
                if line.startswith("checkpoint:"):
                    data['Checkpoint'] = line.split(":", 1)[1].strip()
                elif line.startswith("test_dataset:"):
                    data['Test Dataset'] = line.split(":", 1)[1].strip()
                elif line.startswith("batch_size:"):
                    data['Batch Size'] = line.split(":", 1)[1].strip()
                elif line.startswith("device:"):
                    data['Device'] = line.split(":", 1)[1].strip()
                elif line.startswith("Test Loss:"):
                    parts = line.split(", ")
                    for part in parts:
                        key, value = part.split(":")
                        data[key.strip()] = value.strip()

                # Extract performance metrics (e.g., samples/sec)
                elif "it/s" in line:
                    samples_per_sec = line.split(",")[-1].split("[")[0].strip()
                    data['Samples/Sec'] = samples_per_sec

            # Convert the dictionary to a DataFrame
            return pd.DataFrame([data])
        except Exception as e:
            print(f"Error parsing results.txt file {file_path}: {e}")
            return df_dummy

    @staticmethod
    def parse_res_rvt(file_path):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Parse each line into a dictionary
            data = {}
            for line in lines:
                if line.strip():  # Ignore empty lines
                    key, value = line.strip().split(',', 1)  # Split by the first comma
                    data[key] = float(value)  # Convert value to float

            # Convert the dictionary to a single-row DataFrame
            return pd.DataFrame([data])
        except Exception as e:
            print(f"Error parsing metrics results file {file_path}: {e}")
            return pd.DataFrame()

    def parse_res_timelens(self, file_path):
        try:
            # Register the constructor for tags to ignore
            yaml.add_constructor(None, self.ignore_tags_constructor)

            # https://stackoverflow.com/questions/35968189/retrieving-data-from-a-yaml-file-based-on-a-python-list
            with open(str(file_path), 'r') as r_file:
                data = yaml.load(r_file, Loader=yaml.FullLoader)

            df = pd.DataFrame.from_dict(data)

            dict_seqs_avg = df['per_dataset'].dropna().to_dict()
            dict_total_avg = df['total_average'].dropna().to_dict()
            # Methods and sequences
            seqs = list(dict_seqs_avg.keys())
            methods = list(dict_total_avg.keys())

            res_list = []  # seq, method, PSNR_avg, PSNR_std, SSIM_avg, SSIM_std
            for met in methods:
                for seq in seqs:

                    res_row = [seq, met, self.logs_dict['extra_info'],
                            dict_seqs_avg[seq][met]['PSNR']['mean'], dict_seqs_avg[seq][met]['PSNR']['std'],
                            dict_seqs_avg[seq][met]['SSIM']['mean'], dict_seqs_avg[seq][met]['SSIM']['std']]
                    res_list.append(res_row)

                # Overall results
                res_row = ['Overall', met, self.logs_dict['extra_info'],
                        dict_total_avg[met]['PSNR']['mean'], dict_total_avg[met]['PSNR']['std'],
                        dict_total_avg[met]['SSIM']['mean'], dict_total_avg[met]['SSIM']['std']]
                res_list.append(res_row)

            labels = ["seq", "method", "skip", "PSNR_avg", "PSNR_std", "SSIM_avg", "SSIM_std"]
            df_res = pd.DataFrame(data=res_list, columns=labels).dropna().sort_values(by=['seq', 'method'])

            return df_res

        except Exception as e:
            print(f"Error parsing metrics results file {file_path}: {e}")
            return pd.DataFrame()

    @staticmethod
    def parse_res_esfp(file_path):
        results = {}

        # Read all lines at once
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()

                # Extract sample name, metric, and value
                match = re.match(r".*/test/([^/]+)\s+(\S+)\s+([\d\.eE+-]+)", line)
                if match:
                    sample_name, metric, value = match.groups()
                    
                    # Store data in a dictionary
                    if sample_name not in results:
                        results[sample_name] = {}
                    results[sample_name][metric] = float(value)

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(results, orient="index").reset_index()
        df.rename(columns={"index": "seq"}, inplace=True)
        
        return df

    @staticmethod
    def parse_time(file_path, time_type='infer_time'):
        # Dummy dataframe in case we don't have data
        df_dummy = pd.DataFrame({time_type: [-1]})

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Parse each line in the file
            data = []
            for line in lines:
                if line.strip():  # Ignore empty lines
                    key, value = line.split(':', 1)  # Split into key and value
                    key = key.strip()  # Extract the metric name
                    value = value.strip()  # Extract the value
                    data.append({'Metric': key, time_type: value})

            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error parsing time.txt file {file_path}: {e}")
            return df_dummy

    @staticmethod
    def parse_size(file_path):
        # Dummy dataframe in case we don't have data
        df_dummy = pd.DataFrame({'Size': [-1], 'Path': ['']})

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Parse each line in the file
            data = []
            for line in lines:
                if line.strip():  # Ignore empty lines
                    size, path = line.split('\t', 1)  # Split into size and path
                    size = size.split(':')[1].strip()  # Extract the size value
                    path = path.strip()  # Extract the path
                    data.append({'Size': size, 'Path': path})

            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error parsing size.txt file {file_path}: {e}")
            return df_dummy

    def merge_logs_rpg_replearning(self):
        # Merge the dataframes
        df_res = pd.concat(self.list_df_datasets_logs, axis=1)
        # Keep only the useful columns
        res_metrics = ['Test Accuracy']  # List of metrics
        df_res[res_metrics] = df_res[res_metrics].apply(pd.to_numeric, errors='coerce')
        cols = res_metrics + self.common_metrics
        return df_res[cols]

    def merge_logs_rvt(self):
        # Merge the dataframes
        df_res = pd.concat(self.list_df_datasets_logs, axis=1)
        # Keep only the useful columns
        # ['test-AP', 'test-AP_50', 'test-AP_75', 'test-AP_S', 'test-AP_M', 'test-AP_L', 'Size', 'Path', 'Metric', 'infer_time', 'Metric', 'preproc_time']
        res_metrics = ['test-AP', 'test-AP_50', 'test-AP_75', 'test-AP_S', 'test-AP_M', 'test-AP_L']
        my_metrics = ['preproc_time']
        df_res[res_metrics] = df_res[res_metrics].apply(pd.to_numeric, errors='coerce')
        cols = res_metrics + self.common_metrics + my_metrics
        return df_res[cols]

    def merge_logs_timelens(self):
        # Merge the dataframes
        df_res = pd.concat(self.list_df_datasets_logs, axis=1)
        # Keep only the useful columns
        # ['test-AP', 'test-AP_50', 'test-AP_75', 'test-AP_S', 'test-AP_M', 'test-AP_L', 'Size', 'Path', 'Metric', 'infer_time', 'Metric', 'preproc_time']
        cols_nominal = ['seq']
        cols_numeric = ['PSNR_avg', 'PSNR_std', 'SSIM_avg', 'SSIM_std']
        res_metrics = cols_nominal + cols_numeric
        df_res[cols_numeric] = df_res[cols_numeric].apply(pd.to_numeric, errors='coerce')
        cols = res_metrics + self.common_metrics
        return df_res[cols]

    def merge_logs_esfp(self):
        # Merge the dataframes
        df_res = pd.concat(self.list_df_datasets_logs, axis=1)
        # Keep only the useful columns
        # ['test-AP', 'test-AP_50', 'test-AP_75', 'test-AP_S', 'test-AP_M', 'test-AP_L', 'Size', 'Path', 'Metric', 'infer_time', 'Metric', 'preproc_time']
        cols_nominal = ['seq']
        cols_numeric = ['mean_ae', 'ae_11', 'ae_22', 'ae_30']
        res_metrics = cols_nominal + cols_numeric
        df_res[cols_numeric] = df_res[cols_numeric].apply(pd.to_numeric, errors='coerce')
        cols = res_metrics + self.common_metrics
        return df_res[cols]

    # Define a custom constructor for tags to ignore
    def ignore_tags_constructor(self, loader, node):
        if isinstance(node, yaml.ScalarNode):
            if node.tag in ['!tag', '!tag_to_ignore_2']:
                return None
        return node

    def read_data(self, logs_path):
        # Register the constructor for tags to ignore
        yaml.add_constructor(None, self.ignore_tags_constructor)

        # https://stackoverflow.com/questions/35968189/retrieving-data-from-a-yaml-file-based-on-a-python-list
        with open(str(logs_path), 'r') as r_file:
            data = yaml.load(r_file, Loader=yaml.FullLoader)

        df = pd.DataFrame.from_dict(data)
        return df


class TableGenerator:
    def __init__(self):
        self.keys_res_metrics = None
        # Vars
        self.cols_common = ['dataset', 'sampler', 'run', 'prob_init', 'method', 'exp', 'Path']
        self.cols_numeric = ['Size', 'infer_time']
        # Aux vars
        self.vars_hsergb = None
        # Initialize
        self._init_main_metrics_()
        self._init_aux_vars_()

    def _init_main_metrics_(self):
        self.keys_res_metrics = {
            'rpg_replearning': ['Test Accuracy'],
            'rvt': ['test-AP_50'],
            'timelens': ['PSNR_avg', 'SSIM_avg'],
            'esfp': ['mean_ae', 'ae_11', 'ae_22', 'ae_30']
        }

    def _init_aux_vars_(self):
        # Vars hsergb
        self.vars_hsergb = {
            'close': ['baloon_popping', 'candle', 'confetti', 'fountain_bellevue2', 'fountain_schaffhauserplatz_02', 'spinning_plate', 'spinning_umbrella', 'water_bomb_eth_01', 'water_bomb_floor_01'],
            'far':   ['bridge_lake_01', 'bridge_lake_03', 'kornhausbruecke_letten_random_04', 'lake_01', 'lake_03', 'sihl_03']
        }
        # Vars esfp
        self.vars_esfp = {
            'has_distractor': ['bowl1_31_10', 'bowl3_31_10', 'bowl4_31_10', 'bowl5_31_10', 'comb1_02_11', 'comb3_02_11', 'comb5_02_11', 'comb6_02_11', 'comb9_02_11', 'comb10_02_11', 'cup_new_02_11', 'drink_mate_31_10', 'headphones_case_31_10', 'ladle_31_10', 'ladle2_31_10', 'phone_02_11', 'stapler_02_11', 'steel_1_31_10', 'table_tennis_02_11', 'vase2_31_10']
        }

    def filter_df(self, df, method):
        # Filter the dataframe
        try:
            cols = self.cols_common + self.cols_numeric + self.keys_res_metrics[method]
            df[self.cols_numeric] = df[self.cols_numeric].apply(pd.to_numeric, errors='coerce')
            return df[cols]
        except KeyError:
            return pd.DataFrame()
    
    def filter_distractors_esfp(self, df):
        # Filter the dataframe
        try:
            mask_seqs_with_distractors = df['seq'].isin(self.vars_esfp['has_distractor'])
            return df[~mask_seqs_with_distractors]
        except KeyError:
            return pd.DataFrame()

    def generate_stats(self, df, method):
        # Define aggregation functions, using 'first' for 'Path'
        agg_funcs = {col: ['mean', 'std'] for col in self.cols_numeric + self.keys_res_metrics[method]}
        #agg_funcs = {col: ['max', 'std'] for col in self.cols_numeric + self.keys_res_metrics[method]}  # "Max instead of min"
        agg_funcs['run'] = 'max'  # Take the maximum run per group
        agg_funcs['Path'] = 'first'  # Take the first Path per group

        # Compute statistics
        df_stats = df.groupby(['dataset', 'sampler', 'prob_init', 'method', 'exp']).agg(agg_funcs).fillna(0)

        # Flatten the multi-index columns, keeping 'Path' unchanged
        df_stats.columns = [
            f"{col[0]}-{col[1]}" if isinstance(col, tuple) and col[1] and col[0] != "Path" else col[0] 
            for col in df_stats.columns
        ]

        return df_stats.reset_index()

    def group_logs(self, list_df_datasets_logs):
        list_df = []
        
        for df in list_df_datasets_logs:
            if df.empty:
                continue
            methods = df['method'].unique()
            for method in methods:

                # Timelens is a special case (close and far sequences)
                if method == 'timelens':
                    df_close_far = [df[df['seq'].isin(self.vars_hsergb[key])] for key in self.vars_hsergb.keys()]
                    
                    for key, df_t in zip(self.vars_hsergb.keys(), df_close_far):
                        # Add the key to the dataset name (e.g. hserb-skip_5-close)
                        df_t['dataset'] = df_t['dataset'] + f"-{key}"
                        # Filter the dataframe (locally)
                        df_filtered = self.filter_df(df_t, method)
                        if df_filtered.empty:
                            continue
                        # Append to the list
                        df_stats = self.generate_stats(df_filtered, method)
                        list_df.append(df_stats)

                if method == 'esfp':
                    pass
                    #df = self.filter_distractors_esfp(df)

                # Filter the dataframe (globally)
                df_filtered = self.filter_df(df, method)
                if df_filtered.empty:
                    continue
                # Append to the list
                df_stats = self.generate_stats(df_filtered, method)
                list_df.append(df_stats)

        # Concatenate the dataframes
        df_main = pd.concat(list_df, axis=0, join='outer').reset_index(drop=True)
        return df_main

    def generate_side_by_side_latex_table(self, df):
        # Generate the latex table
        datasets = df['dataset'].unique()
        list_df_datasets = []
        for dataset in datasets:
            df_dataset = df[df['dataset'] == dataset]
            df_dataset = df_dataset.dropna(axis=1, how='all')
            list_df_datasets.append(df_dataset)

        df_latex = pd.concat(list_df_datasets, axis=1)
        return df_latex

    def generate_main_table(self, list_df_datasets_logs):
        # Generate main table
        df_main = self.group_logs(list_df_datasets_logs)
        # Generate the latex table
        ########
        # Dataset1                                    Dataset2                                    ...
        # Method Prob Metric(Mean) Metric(std) Size   Method Prob Metric(Mean) Metric(std) Size   ...
        ########

        # Generate the latex table
        df_latex = self.generate_side_by_side_latex_table(df_main)
        print(tabulate(df_latex))

        return df_main

    def generate_data_files(self, list_df_datasets_logs):
        # Generate main table
        df_main = self.group_logs(list_df_datasets_logs)
        # Generate data files for tables and plots
        logs_path = Path(f"{cfg_master.common.docker_app}/{cfg_master.common.logs_path}")
        logs_data_path = logs_path / 'data'
        logs_data_path.mkdir(parents=True, exist_ok=True)

        # Load mappings from the configuration file
        sampler_map_path = Path(cfg_master.common.docker_app) / Path(cfg_master.common.cfg_root) / Path(cfg_master.common.csv_sampler_map)
        sampler_map = pd.read_csv(sampler_map_path, index_col="sampler").to_dict()["sampler_id"]

        # Define fixed columns
        fixed_cols = ["dataset", "sampler", "sampler_id", "prob_init", "run-max", "method", "exp", "infer_time-mean", "Size-mean", "Path"]
        #fixed_cols = ["dataset", "sampler", "sampler_id", "prob_init", "run-max", "method", "exp", "infer_time-max", "Size-max", "Path"]

        dataset_list = df_main['dataset'].unique()
        # Assign the sampler id
        df_main = assign_sampler_id(df_main, sampler_map)

        for dataset in dataset_list:
            df_dataset = df_main[df_main['dataset'] == dataset]
            var_cols = [col for col in df_dataset.columns if col not in fixed_cols]
            # Drop NaN columns only in var_cols
            df_dataset = df_dataset.drop(columns=[col for col in var_cols if df_dataset[col].isna().all()])
            var_cols = [col for col in df_dataset.columns if col not in fixed_cols]
            cols = fixed_cols + var_cols
            df_dataset = df_dataset[cols]
            df_dataset = df_dataset.fillna(-1)

            if cfg_poled.oled.flag_train:
                df_dataset.to_csv(f"{logs_data_path}/{dataset}_retrained_data.csv", index=False)
            else:   
                df_dataset.to_csv(f"{logs_data_path}/{dataset}_data.csv", index=False)


def bash_array_to_list(bash_array):
    return shlex.split(bash_array.strip('()'))


def generate_logs_df(res_parser, logs_dict):
    # Parse the logs
    df_logs = res_parser.parse_logs(logs_dict)

    return df_logs


def check_errors(df):
    # Some error has occurred, do not append to the list
    if df is None:
        return -1
    if isinstance(df, int) and df == -1:
        return -1
    if df.empty:
        return -1

    return df


def add_info_to_logs_df(df, **kwargs):
    # Add information to the dataframe
    df_res = df.copy()
    
    df_res['dataset'] = kwargs.get('dataset', '-1')
    df_res['sampler'] = kwargs.get('sampler', '-1')
    df_res['run'] = kwargs.get('run', '-1')
    df_res['prob_init'] = kwargs.get('prob_init', '-1')
    df_res['method'] = kwargs.get('method', '-1')
    df_res['exp'] = kwargs.get('exp', '-1')

    return df_res


def parse_logs(args_dict):
    # cfgs in oled
    oled_cfg_list = bash_array_to_list(cfg_poled.oled.exp_cfgs)
    # datasets in oled
    oled_datasets_list = bash_array_to_list(cfg_poled.oled.datasets_name)
    # samplers in oled
    oled_samplers_list = bash_array_to_list(cfg_poled.oled.samplers)
    # runs in oled
    oled_runs_list = bash_array_to_list(cfg_poled.oled.runs)
    # prob_inits in oled
    oled_prob_inits_list = bash_array_to_list(cfg_poled.oled.prob_inits)

    # 1. After running "run_experiments.sh" we want to generate the tables and figures
    # Caveat: Methods are specificed in "results.yaml"
    
    # Generate the path to the logs
    logs_path = Path(f"{cfg_master.common.docker_app}/{cfg_master.common.logs_path}")

    # Initialize the parser
    res_parser = ResultsParser()

    # Log dataframes
    list_df_dataset_logs = []

    for exp_yaml in oled_cfg_list:
        # Load experiment configuration that was saved to OLED logs
        #logs_sys_path="$common_root"/"$common_logs_path"/"$dataset_name"/oled/"$exp_name"/"$run_descriptor"

        cfg_exp = ut.load_yaml(f"{cfg_master.common.docker_app}/{cfg_master.common.cfg_root}/{cfg_master.common.cfg_sampling}/{exp_yaml}")
        for dataset in oled_datasets_list:
            list_df_logs = []
            for sampler in oled_samplers_list:
                for run in oled_runs_list:
                    for prob_init_exp in oled_prob_inits_list:
                        # Generate the run descriptor
                        prob_init = float(f"0.{prob_init_exp}") if float(prob_init_exp) < 10 else 1
                        run_descriptor = f"{sampler}_{prob_init}-{run}"   

                        # Dataset logs path
                        logs_dataset_path = logs_path / dataset
                        # OLED logs
                        logs_oled_path = Path(f"{logs_dataset_path}/oled/{cfg_exp.sampler_params.exp_name}/{run_descriptor}")
                        # Check if the downsampling has been done
                        if not logs_oled_path.exists():
                            continue

                        # Methods inside the dataset
                        methods_dataset = [method.name for method in logs_dataset_path.iterdir() if method.is_dir() and method.name in cfg_res.experiment.methods]
                        
                        for method in methods_dataset:
                            logs_exp_path = Path(f"{logs_path}/{dataset}/{method}/{cfg_exp.sampler_params.exp_name}")
                            # Extra information for some methods is within exp_name and run_descriptor (e.g. timelens has different skips)
                            # E.g. .../cvpr25/det_0.1-1
                            # or
                            # E.g. .../cvpr25/skip_5/det_0.1-1
                            extra_info = [e.name for e in logs_exp_path.iterdir() if e.is_dir()]

                            if run_descriptor not in extra_info:
                                for e in extra_info:
                                    logs_method_path = Path(f"{logs_path}/{dataset}/{method}/{cfg_exp.sampler_params.exp_name}/{e}/{run_descriptor}")

                                    # Log files to dictionary
                                    logs_dict = {
                                        'method': method,
                                        'log_results': logs_method_path / cfg_master.common.logs_res_file,
                                        'log_size': logs_method_path / cfg_master.common.logs_size_file,
                                        'log_time': logs_method_path / cfg_master.common.logs_time_file,
                                        'log_sampling_time': logs_oled_path / cfg_master.common.logs_sampling_time_file,
                                        'log_preproc_time': logs_method_path / cfg_master.common.logs_preproc_time_file,
                                        'extra_info': e
                                    }

                                    if not logs_dict['log_results'].exists():
                                        continue

                                    # Generate the logs dataframe
                                    df_logs = generate_logs_df(res_parser, logs_dict)

                                    # Check if there are errors
                                    df_logs = check_errors(df_logs)

                                    if isinstance(df_logs, int) and df_logs == -1:
                                        continue

                                    # Add information to the dataframe
                                    info_dict = {
                                        "dataset": f"{dataset}-{e}",
                                        "sampler": sampler,
                                        "run": run,
                                        "prob_init": prob_init,
                                        "method": method,
                                        "exp": cfg_exp.sampler_params.exp_name
                                    }
 
                                    df_logs = add_info_to_logs_df(df_logs, **info_dict)
                                    list_df_logs.append(df_logs)
                                    df_logs = None

                            else:
                                logs_method_path = Path(f"{logs_path}/{dataset}/{method}/{cfg_exp.sampler_params.exp_name}/{run_descriptor}")

                                if cfg_poled.oled.flag_train:
                                    logs_method_path = logs_method_path / cfg_master.common.logs_retrained_path

                                # Log files to dictionary
                                logs_dict = {
                                    'method': method,
                                    'log_results': logs_method_path / cfg_master.common.logs_res_file,
                                    'log_size': logs_method_path / cfg_master.common.logs_size_file,
                                    'log_time': logs_method_path / cfg_master.common.logs_time_file,
                                    'log_sampling_time': logs_oled_path / cfg_master.common.logs_sampling_time_file,
                                    'log_preproc_time': logs_method_path / cfg_master.common.logs_preproc_time_file
                                }

                                if not logs_dict['log_results'].exists():
                                    continue

                                # Generate the logs dataframe
                                df_logs = generate_logs_df(res_parser, logs_dict)

                                # Check if there are errors
                                df_logs = check_errors(df_logs)

                                if isinstance(df_logs, int) and df_logs == -1:
                                    continue

                                # Add information to the dataframe
                                info_dict = {
                                    "dataset": f"{dataset}",
                                    "sampler": sampler,
                                    "run": run,
                                    "prob_init": prob_init,
                                    "method": method,
                                    "exp": cfg_exp.sampler_params.exp_name
                                }

                                df_logs = add_info_to_logs_df(df_logs, **info_dict)
                                list_df_logs.append(df_logs)
                                df_logs = None

            # Concatenate the dataframes
            if len(list_df_logs) < 1:
                continue

            df_logs_dataset = pd.concat(list_df_logs)
            list_df_dataset_logs.append(df_logs_dataset)

    # 2 Save the main table to the logs folder
    # Generate Latex tables (this has to go inside the latex class)
    table_generator = TableGenerator()
    df_latex = table_generator.generate_main_table(list_df_dataset_logs)
    custom_sampler_order = ['det', 'uniform', 'poisson', 'evDownNavi']
    df_latex['sampler'] = pd.Categorical(df_latex['sampler'], categories=custom_sampler_order, ordered=True)
    df_latex = df_latex.sort_values(by=['dataset', 'prob_init', 'method', 'sampler', 'exp'])

    # Save the table to the logs directory
    res_path = logs_path / 'res_table.txt'
    with open(res_path, 'w') as f:
        f.write(tabulate(df_latex))
        #f.write(tabulate(df_latex, headers=df_latex.columns))

    return list_df_dataset_logs


def assign_sampler_id(df, sampler_map):
    # Assign the sampler id, ensure all samplers have a unique id (even if not in the mapping)
    df_res = df.copy()
    sampler_map_res = sampler_map.copy()

    max_id = max(sampler_map.values(), default=0)
    for sampler in df['sampler'].unique():
        if sampler not in sampler_map:
            max_id += 1
            sampler_map_res[sampler] = max_id

    df_res['sampler_id'] = df['sampler'].map(sampler_map_res)
    return df_res


def main(args_dict):
    # Parse the logs
    list_df_datasets_logs = parse_logs(args_dict)
    # Generate tables
    table_generator = TableGenerator()
    # Generate data files for tables and plots
    table_generator.generate_data_files(list_df_datasets_logs)
    # Generate readable table
    df_latex = table_generator.generate_main_table(list_df_datasets_logs)


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Parse the results of the experiments and generate the tables and figures')
    parser.add_argument('--cfg_res', type=str, default='/home/user/app/config/experiments/basic.yaml')
    parser.add_argument('--cfg_master', type=str, default='/home/user/app/config/master.yaml')
    parser.add_argument('--cfg_poled', type=str, default='/home/user/app/config/ev_sampling/poled.yaml')
    parser.add_argument('--output_dir', type=str, help='Directory to save the tables and figures')
    args = parser.parse_args()

    args_dict={
        'cfg_res': args.cfg_res,
        'cfg_master': args.cfg_master,
        'cfg_poled': args.cfg_poled,
        'output_dir': args.output_dir
    }

    # Load the configuration file
    cfg_res = ut.load_yaml(args_dict['cfg_res'])
    cfg_master = ut.load_yaml(args_dict['cfg_master'])
    cfg_poled = ut.load_yaml(args_dict['cfg_poled'])

    main(args_dict)