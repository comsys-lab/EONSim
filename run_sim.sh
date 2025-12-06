#!/bin/bash

### outdir ### 
OUT="results"
mkdir -p $OUT
##############

### workload & dataset ###
workload_path_dir="$(pwd)/workload_configs/"
workload="dlrm_rmc2_small"
WORKLOAD_CONFIG="${workload_path_dir}${workload}"

data_path_dir="$(pwd)/datasets/"
dataset_list=("dlrm/reuse_high_test.txt")
###############

### simulation parameters ###
MEM_CFG=$1
MATRIX_CFG="tpuv6e.cfg"

NUM_BATCH=1
BS=32

# Set PROF_MULTIPLIER to $2 if provided, otherwise default to 1
PROF_MULTIPLIER=${2:-1}
##############################

### others ###
WORKLOAD_NAME=$workload
# Generate random number for unique temporary directory
RAND_NUM=$(python3 -c "import random; print(random.randint(1, 100000000))")
OUTDIR="${OUT}/${WORKLOAD_NAME}_${NUM_BATCH}_${BS}_${RAND_NUM}"
echo "Output Directory for Logs: $OUTDIR"
mkdir -p $OUTDIR
##############

for dataset in "${dataset_list[@]}"; do
    DATA_GEN_PATH=$data_path_dir$dataset
    
    # Filename generation for output
    DATASET_STR=$(echo "$dataset" | sed 's/\//_/g')
    OUTPUT_FILENAME="${DATASET_STR}_${MEM_CFG}_${NUM_BATCH}batch"
    
    # Log file path
    LOGFILE="$OUTDIR/$OUTPUT_FILENAME.log"
    
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
        
    # Move log file to the output directory determined by python script
    if [ -f .last_output_dir ]; then
        TARGET_DIR=$(cat .last_output_dir)
        if [ -d "$TARGET_DIR" ]; then
            echo "Moving log file to $TARGET_DIR"
            mv $LOGFILE $TARGET_DIR/
        fi
        rm .last_output_dir
    fi
done

# Remove the temporary output directory if it is empty
if [ -d "$OUTDIR" ]; then
    if [ -z "$(ls -A $OUTDIR)" ]; then
        rmdir $OUTDIR
        echo "Removed empty temporary directory: $OUTDIR"
    fi
fi
