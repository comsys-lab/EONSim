#!/bin/bash
# Original code: https://github.com/rishucoding/reproduce_MICRO24_GPU_DLRM_inference

### outdir ### 
OUT="results_onmem"
mkdir -p $OUT
##############

### dataset ###
data_path_dir="$(pwd)/datasets/"
# dataset_list=("dlrm/reuse_low_trunc.txt") # This line is no longer needed.
###############

### simulation parameters ###
MEM_CFG=$1 # spad_naive
EMB_DIM=128
EMB_ROW=100000000 #1000000
EMB_TBL=1 # 512
EMB_POOL=1000000
EMBS="$EMB_DIM,$EMB_ROW,$EMB_TBL,$EMB_POOL"
NUM_FORMAT=8

NUM_BATCH=1
BS=1

# Set PROF_MULTIPLIER to $2 if provided, otherwise default to 1
PROF_MULTIPLIER=${2:-1}
##############################

### mNPUsim-related parameters ###
OFFMEM_CFG="dram_config/total_dram_config/single_hbm3_819gbs.cfg"
NPUMEM_CFG="npumem_config/npumem_architecture_list/single.txt"
##################################

### others ###
PyGenTbl='import sys; rows,tables=sys.argv[1:3]; print("-".join([rows]*int(tables)))'
OUTDIR="$(echo "$EMBS" | sed 's/,/_/g')"
echo $OUTDIR
OUTDIR="${OUT}/${OUTDIR}_${NUM_BATCH}_${BS}"
echo $OUTDIR
mkdir -p $OUTDIR
##############

# Loop from 0 to 999 to generate dataset paths dynamically
for i in {9..9}; do
    # Construct the dataset file path for the current iteration
    dataset="random/random_trace_${i}.txt"

    DATA_GEN_PATH=$data_path_dir$dataset
    OUTFILE=$(echo "$dataset" | sed 's/\//_/g')
    OUTFILE="$OUTDIR/$OUTFILE"
    for e in $EMBS; do
        IFS=','; set -- $e; EMB_DIM=$1; EMB_ROW=$2; EMB_TBL=$3; EMB_LS=$4; unset IFS;
        EMB_TBL=$(python3 -c "$PyGenTbl" "$EMB_ROW" "$EMB_TBL")
        python3 src/simulator.py --num-batches $NUM_BATCH --batch-size $BS --numeric-format-bits $NUM_FORMAT\
            --lookups-per-sample $EMB_LS --arch-sparse-feature-size $EMB_DIM\
            --arch-embedding-size $EMB_TBL --data-generation=$DATA_GEN_PATH --memory-config=$MEM_CFG\
            --offchip-memory-config=$OFFMEM_CFG --npumem-config=$NPUMEM_CFG\
            --profiling-multiplier $PROF_MULTIPLIER | tee $(pwd)/${OUTFILE}_${MEM_CFG}_${PROF_MULTIPLIER}batch.log
    done
done