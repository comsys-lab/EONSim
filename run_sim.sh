#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <memory_config>"
    exit 1
fi

### outdir ###
OUT="results"
mkdir -p "$OUT"
##############

### workload & dataset ###
workload_path_dir="$(pwd)/workload_configs/"
workload="dlrm_dcnv2"
WORKLOAD_CONFIG="${workload_path_dir}${workload}"

data_path_dir="$(pwd)/datasets/"
dataset_list=("dlrm/reuse_high_test.txt") # "dlrm/reuse_high_test.txt" "dlrm/reuse_medium_test.txt" "dlrm/reuse_low_trunc.txt"
###############

### simulation parameters ###
MEM_CFG=$1

NUM_BATCH=1
BS=256

# EONSim skips memory simulation for the warmup batches to speed up the simulation. 
# Temporarily set all batches except for the last batch as warmup for faster simulation and testing.
WARMUP_BATCHES=$((NUM_BATCH - 1))

# Set PROF_PERIOD to $2 if provided, otherwise default to 1.
# This parameter is to set the profile period for profiling-based policies. For the other policies, this parameter will be ignored.
# For profile_dynamic_* policies, this is the batch-granularity period for profiling refresh:
#   1 -> refresh every batch, 2 -> every 2 batches, N -> every N batches.
# Example: ./run_sim.sh profile_dcache 4
PROF_PERIOD=${2:-1}
##############################

for dataset in "${dataset_list[@]}"; do
    DATA_GEN_PATH=$data_path_dir$dataset
    
    # Resolve target base output directory before running simulator.
    BASE_TARGET_DIR=$(python3 src/helper_modules/helper.py \
        --resolve-output-dir \
        --workload-config "$WORKLOAD_CONFIG" \
        --output-base-dir "$OUT" \
        --batch-size "$BS" \
        --dataset-path "$DATA_GEN_PATH")

    # Add config folder to target dir
    CONFIG_NAME=$(basename "$MEM_CFG")
    TARGET_DIR="$BASE_TARGET_DIR/$CONFIG_NAME"
    mkdir -p "$TARGET_DIR"

    # Define suffix based on policy type
    if [[ "$MEM_CFG" == profile* ]]; then
        SUFFIX="_${PROF_PERIOD}Bperiod"
    else
        SUFFIX=""
    fi

    # Log file path
    LOGFILE="$TARGET_DIR/sim_log${SUFFIX}.log"
    
    # Pass suffix to simulator as output_filename so it can append to emb_results and matrix_results
    OUTPUT_FILENAME="${SUFFIX}"
    
    echo "Running simulation for dataset: $dataset with config base: $WORKLOAD_CONFIG"
    
    python3 src/simulator.py \
        --workload-config $WORKLOAD_CONFIG \
        --num-batches $NUM_BATCH \
        --batch-size $BS \
        --data-generation=$DATA_GEN_PATH \
        --memory-config=$MEM_CFG \
        --profiling-period $PROF_PERIOD \
        --warmup-batches $WARMUP_BATCHES \
        --output-filename "$OUTPUT_FILENAME" \
        | tee "$LOGFILE"
done
