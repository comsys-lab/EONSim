#!/bin/bash
# Original code: https://github.com/rishucoding/reproduce_MICRO24_GPU_DLRM_inference

### outdir ### 
OUT="results_val_forv6e_tables"
mkdir -p $OUT
##############

### dataset ###
data_path_dir="$(pwd)/datasets/"
dataset_list=("dlrm/reuse_low_trunc.txt")
###############

### simulation parameters ###
MEM_CFG=$1 # spad_naive
EMB_DIM=128
EMB_ROW=500000
EMB_POOL=150
NUM_FORMAT=32

NUM_BATCH=1  # Fixed number of batches
BATCH_SIZE=256  # Fixed batch size

# EMB_TBL iteration parameters
TBL_START=${2:-10}       # Starting number of tables (default 10)
TBL_STEP=${3:-10}        # Step size for table increment (default 10)
TBL_MAX=${4:-50}         # Maximum number of tables (default 50)

# Set PROF_MULTIPLIER to $5 if provided, otherwise default to 1
PROF_MULTIPLIER=${5:-1}
##############################

### mNPUsim-related parameters ###
OFFMEM_CFG="dram_config/total_dram_config/single_hbm3_819gbs.cfg"
NPUMEM_CFG="npumem_config/npumem_architecture_list/single.txt"
##################################

### others ###
PyGenTbl='import sys; rows,tables=sys.argv[1:3]; print("-".join([rows]*int(tables)))'
##############

echo "Starting DLRM validation with table count variation..."
echo "Memory config: $MEM_CFG"
echo "Table count range: $TBL_START to $TBL_MAX (step: $TBL_STEP)"
echo "Fixed batch size: $BATCH_SIZE"
echo "Profiling multiplier: $PROF_MULTIPLIER"
echo ""

for dataset in "${dataset_list[@]}"; do
    DATA_GEN_PATH=$data_path_dir$dataset
    OUTFILE_BASE=$(echo "$dataset" | sed 's/\//_/g')
    
    # Iterate through table counts from TBL_START to TBL_MAX
    for (( EMB_TBL=TBL_START; EMB_TBL<=TBL_MAX; EMB_TBL+=TBL_STEP )); do
        EMBS="$EMB_DIM,$EMB_ROW,$EMB_TBL,$EMB_POOL"
        
        OUTDIR="$(echo "$EMBS" | sed 's/,/_/g')"
        OUTDIR="${OUT}/${OUTDIR}_${NUM_BATCH}_${BATCH_SIZE}"
        echo "Creating directory: $OUTDIR"
        mkdir -p $OUTDIR
        
        OUTFILE="$OUTDIR/$OUTFILE_BASE"
        
        for e in $EMBS; do
            IFS=','; set -- $e; EMB_DIM=$1; EMB_ROW=$2; EMB_TBL=$3; EMB_LS=$4; unset IFS;
            EMB_TBL_FORMATTED=$(python3 -c "$PyGenTbl" "$EMB_ROW" "$EMB_TBL")
            echo "Running simulation with EMB_TBL=$EMB_TBL (BATCH_SIZE=$BATCH_SIZE)"
            python3 src/simulator.py --num-batches $NUM_BATCH --batch-size $BATCH_SIZE --numeric-format-bits $NUM_FORMAT\
                --lookups-per-sample $EMB_LS --arch-sparse-feature-size $EMB_DIM\
                --arch-embedding-size $EMB_TBL_FORMATTED --data-generation=$DATA_GEN_PATH --memory-config=$MEM_CFG\
                --offchip-memory-config=$OFFMEM_CFG --npumem-config=$NPUMEM_CFG\
                --profiling-multiplier $PROF_MULTIPLIER | tee $(pwd)/${OUTFILE}_${MEM_CFG}_${PROF_MULTIPLIER}batch.log
        done
    done
done

echo ""
echo "All simulations completed!"
echo "Usage: ./dlrm_validation_tables.sh [MEM_CFG] [TBL_START] [TBL_STEP] [TBL_MAX] [PROF_MULTIPLIER]"
echo "Example: ./dlrm_validation_tables.sh spad_off 10 10 50 1"