#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <memory_config>"
    exit 1
fi

### USER CONFIGURATION ####################################
WORKLOAD="dlrm_dcnv2"
DATASET_LIST=("dlrm/reuse_high_test.txt") 

NUM_BATCH=5
BS=256

# EONSim skips memory simulation for the warmup batches to speed up the simulation.
# Temporarily set all batches except for the last batch as warmup for faster simulation and testing.
WARMUP_BATCHES=$((NUM_BATCH - 1))

OUT="results"
DEBUG=0  # Set to 1 to enable debug output
###########################################################

### CLI ARGUMENTS #########################################
# $1 (required): memory_config  e.g. cache_LRU
# $2 (optional): profiling_period (default: 1)
# For profile_dynamic_* policies, this is the batch-granularity period for profiling refresh:
#   1 -> refresh every batch, 2 -> every 2 batches, N -> every N batches.
# Example: ./run_sim.sh profile_dcache 4
MEM_CFG=$1
PROF_PERIOD=${2:-1}
###########################################################






# --- derived / internal — do not edit below ---
WORKLOAD_PATH_DIR="$(pwd)/workload_configs/"
WORKLOAD_CONFIG="${WORKLOAD_PATH_DIR}${WORKLOAD}"

DATA_PATH_DIR="$(pwd)/datasets/"

mkdir -p "$OUT"

# Translate DEBUG flag to a simulator option string
DEBUG_FLAG=""
if [ "$DEBUG" -eq 1 ]; then
    DEBUG_FLAG="--debug"
fi

for DATASET in "${DATASET_LIST[@]}"; do
    DATA_GEN_PATH=${DATA_PATH_DIR}${DATASET}

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

    echo "Running simulation for dataset: $DATASET with config base: $WORKLOAD_CONFIG"

    python3 src/simulator.py \
        --workload-config $WORKLOAD_CONFIG \
        --num-batches $NUM_BATCH \
        --batch-size $BS \
        --data-generation=$DATA_GEN_PATH \
        --memory-config=$MEM_CFG \
        --profiling-period $PROF_PERIOD \
        --warmup-batches $WARMUP_BATCHES \
        --output-filename "$OUTPUT_FILENAME" \
        $DEBUG_FLAG \
        | tee "$LOGFILE"
done
