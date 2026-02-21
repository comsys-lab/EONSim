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
dataset_list=("dlrm/reuse_high_test.txt")
###############

### simulation parameters ###
MEM_CFG=$1
MATRIX_CFG="tpuv6e.cfg"

NUM_BATCH=1
BS=256

# Set PROF_MULTIPLIER to $2 if provided, otherwise default to 1 (this is only used in spad_oracle config)
PROF_MULTIPLIER=${2:-1}
##############################

for dataset in "${dataset_list[@]}"; do
    DATA_GEN_PATH=$data_path_dir$dataset
    
    # Filename generation for output
    DATASET_STR=$(echo "$dataset" | sed 's/\//_/g')
    OUTPUT_FILENAME="${DATASET_STR}_${MEM_CFG}_${NUM_BATCH}batch"
    
    # Resolve target output directory before running simulator.
    TARGET_DIR=$(python3 src/helper_modules/Helper.py \
        --resolve-output-dir \
        --workload-config "$WORKLOAD_CONFIG" \
        --output-base-dir "$OUT" \
        --batch-size "$BS")

    mkdir -p "$TARGET_DIR"

    # Log file path
    LOGFILE="$TARGET_DIR/$OUTPUT_FILENAME.log"
    
    echo "Running simulation for dataset: $dataset with config base: $WORKLOAD_CONFIG"
    
    python3 src/simulator.py \
        --workload-config $WORKLOAD_CONFIG \
        --num-batches $NUM_BATCH \
        --batch-size $BS \
        --data-generation=$DATA_GEN_PATH \
        --memory-config=$MEM_CFG \
        --matrix-config=$MATRIX_CFG \
        --profiling-multiplier $PROF_MULTIPLIER \
        --output-filename $OUTPUT_FILENAME \
        | tee $LOGFILE
done
