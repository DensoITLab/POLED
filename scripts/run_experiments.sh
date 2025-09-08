# Run all the pipeline

root_path=$POLED_PATH
echo "Root path:" "$root_path"

# Parse YAML function
source "$root_path"/scripts/parse_yaml.sh
# Parse the master YAML file
eval $(parse_yaml "$root_path"/config/master.yaml)
# Parse datasets paths YAML file (datasets)
datasets_yaml="$common_root"/"$common_cfg_root"/"$common_cfg_datasets"/"$datasets_cfg_file"
eval $(parse_yaml "$datasets_yaml")
# Parse the POLED YAML file
poled_yaml="$common_root"/"$common_cfg_root"/"$common_cfg_sampling"/"$poled_cfg_file"
eval $(parse_yaml "$poled_yaml")

# Run POLED
if [[ "$experiments_flag_downsample" == "true" ]]; then
    echo "Running POLED..."
    bash "$root_path"/scripts/run_POLED.sh
fi

# Run Methods
if [[ "$experiments_flag_methods" == "true" ]]; then
    if [[ "$poled_flag_train" == "true" ]]; then
        # Run Representation Learning
        if [[ " ${poled_datasets_name[*]} " =~ " NCaltech101 " ]]; then
            echo "Running Representation Learning in training mode..."
            bash "$root_path"/scripts/run_rep_learning_training.sh
        fi
        
        # Run ESFP
        if [[ " ${poled_datasets_name[*]} " =~ " esfp " ]]; then
            echo "Running ESFP training..."
            bash "$root_path"/scripts/run_ESFP_training.sh
        fi
    else
        # Run Representation Learning
        if [[ " ${poled_datasets_name[*]} " =~ " NCaltech101 " ]]; then
            echo "Running Representation Learning in testing mode..."
            bash "$root_path"/scripts/run_rep_learning.sh
        fi

        # Run RVT
        if [[ " ${poled_datasets_name[*]} " =~ " gen1 " ]]; then
            echo "Running RVT..."
            bash "$root_path"/scripts/run_RVT.sh
        fi

        # Run timelens
        if [[ " ${poled_datasets_name[*]} " =~ " hsergb " ]]; then
            echo "Running Timelens..."
            bash "$root_path"/scripts/run_timelens.sh
        fi

        # Run ESFP
        if [[ " ${poled_datasets_name[*]} " =~ " esfp " ]]; then
            echo "Running ESFP..."
            bash "$root_path"/scripts/run_ESFP.sh
        fi
    fi
fi

# Run logs parser
if [[ "$experiments_flag_results" == "true" ]]; then
    echo "Running logs parser..."
    docker exec -it "$poled_docker_id" python "$poled_docker_app"/src/parse_results.py
fi