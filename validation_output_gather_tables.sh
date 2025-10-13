#!/bin/bash

# Script to gather Off-chip Memory Cycles from simulation logs into CSV format
# Based on table count iteration

### Parameters ###
OUT_DIR="results_val_forv6e_tables_b256"  # Base output directory (should match dlrm_validation_tables.sh)
MEM_CFG=${1:-"spad_off"}             # Memory configuration (first argument)
TBL_START=${2:-10}                   # Starting number of tables (second argument)
TBL_STEP=${3:-10}                    # Step size for table increment (third argument)
TBL_MAX=${4:-50}                     # Maximum number of tables (fourth argument)
PROF_MULTIPLIER=${5:-1}              # Profiling multiplier (fifth argument)

# Fixed simulation parameters (should match dlrm_validation_tables.sh)
EMB_DIM=128
EMB_ROW=500000
EMB_POOL=150
NUM_BATCH=1                          # Fixed number of batches
BATCH_SIZE=256                       # Fixed batch size
NUM_FORMAT=32
##################

# Dataset info
DATASET_FILE="dlrm_reuse_low_trunc.txt"

# Output CSV file
CSV_OUTPUT="off_chip_memory_cycles_${MEM_CFG}_${PROF_MULTIPLIER}_tables.csv"

echo "Gathering Off-chip Memory Cycles data based on table count..."
echo "Memory config: $MEM_CFG"
echo "Table count range: $TBL_START to $TBL_MAX (step: $TBL_STEP)"
echo "Fixed batch size: $BATCH_SIZE"
echo "CSV output: $CSV_OUTPUT"
echo ""

# Create CSV header
echo "Table_Count,Off_chip_Memory_Cycles" > $CSV_OUTPUT

# Iterate through table counts
for (( EMB_TBL=TBL_START; EMB_TBL<=TBL_MAX; EMB_TBL+=TBL_STEP )); do
    # Construct directory path following the same rule as dlrm_validation_tables.sh
    OUTDIR="${EMB_DIM}_${EMB_ROW}_${EMB_TBL}_${EMB_POOL}_${NUM_BATCH}_${BATCH_SIZE}"
    echo "DEBUG: Expected directory: $OUTDIR"
    FULL_OUTDIR="${OUT_DIR}/${OUTDIR}"
    
    # Construct log file path
    LOG_FILE="${FULL_OUTDIR}/${DATASET_FILE}_${MEM_CFG}_${PROF_MULTIPLIER}batch.log"
    
    if [ -f "$LOG_FILE" ]; then
        # Extract Off-chip Memory Cycles value
        CYCLES=$(grep "Off-chip Memory Cycles:" "$LOG_FILE" | grep -o '[0-9]\+' | head -1)
        
        if [ -n "$CYCLES" ] && [ "$CYCLES" != "" ]; then
            echo "$EMB_TBL,$CYCLES" >> $CSV_OUTPUT
            echo "  Table count $EMB_TBL: $CYCLES cycles"
        else
            echo "$EMB_TBL,N/A" >> $CSV_OUTPUT
            echo "  Table count $EMB_TBL: Could not extract cycles"
        fi
    else
        echo "$EMB_TBL,N/A" >> $CSV_OUTPUT
        echo "  Table count $EMB_TBL: Log file not found"
    fi
done

echo ""
echo "Results saved to: $CSV_OUTPUT"
echo "Summary:"
cat $CSV_OUTPUT

echo ""
echo "Usage: ./validation_tables_output_gather.sh [MEM_CFG] [TBL_START] [TBL_STEP] [TBL_MAX] [PROF_MULTIPLIER]"
echo "Example: ./validation_tables_output_gather.sh spad_off 10 10 50 1"