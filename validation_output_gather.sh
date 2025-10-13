#!/bin/bash

# Script to gather Off-chip Memory Cycles from simulation logs into CSV format
# Based on batch size iteration

### Parameters ###
OUT_DIR="results_val_forv6e"  # Base output directory (should match dlrm_validation_batches.sh)
MEM_CFG=${1:-"spad_off"}           # Memory configuration (first argument)
BS_INCREMENT=${2:-8}               # N: batch size increment (second argument)
MAX_BS=${3:-32}                    # M: maximum batch size (third argument)
PROF_MULTIPLIER=${4:-1}            # Profiling multiplier (fourth argument)

# Fixed simulation parameters (should match dlrm_validation_batches.sh)
EMB_DIM=128
EMB_ROW=500000
EMB_TBL=30
EMB_POOL=150
NUM_BATCH=1                        # Fixed number of batches
NUM_FORMAT=32
##################

# Dataset info
DATASET_FILE="dlrm_reuse_low_trunc.txt"

# Output CSV file
CSV_OUTPUT="off_chip_memory_cycles_${MEM_CFG}_${PROF_MULTIPLIER}_batchsize.csv"

echo "Gathering Off-chip Memory Cycles data based on batch size..."
echo "Memory config: $MEM_CFG"
echo "Batch size range: $BS_INCREMENT to $MAX_BS (increment: $BS_INCREMENT)"
echo "CSV output: $CSV_OUTPUT"
echo ""

# Create CSV header
echo "Batch_Size,Off_chip_Memory_Cycles" > $CSV_OUTPUT

# Iterate through batch sizes
for (( BS=BS_INCREMENT; BS<=MAX_BS; BS+=BS_INCREMENT )); do
    # Construct directory path following the same rule as dlrm_validation_batches.sh
    OUTDIR="${EMB_DIM}_${EMB_ROW}_${EMB_TBL}_${EMB_POOL}_${NUM_BATCH}_${BS}"
    FULL_OUTDIR="${OUT_DIR}/${OUTDIR}"
    
    # Construct log file path
    LOG_FILE="${FULL_OUTDIR}/${DATASET_FILE}_${MEM_CFG}_${PROF_MULTIPLIER}batch.log"
    
    echo "DEBUG: Expected directory: $FULL_OUTDIR"
    
    # Check if directory exists
    if [ -d "$FULL_OUTDIR" ]; then
        echo "DEBUG: Directory exists: $FULL_OUTDIR"
        echo "DEBUG: Contents of directory:"
        ls -la "$FULL_OUTDIR"
    else
        echo "DEBUG: Directory does not exist: $FULL_OUTDIR"
    fi
    
    echo "DEBUG: Expected log file: $LOG_FILE"
    
    if [ -f "$LOG_FILE" ]; then
        echo "DEBUG: Log file exists: $LOG_FILE"
        echo "DEBUG: File size: $(stat -c%s "$LOG_FILE") bytes"
        
        # Show lines containing "Off-chip Memory Cycles" for debugging
        echo "DEBUG: Lines containing 'Off-chip Memory Cycles':"
        grep "Off-chip Memory Cycles" "$LOG_FILE" || echo "DEBUG: No lines found with 'Off-chip Memory Cycles'"
        
        # Extract Off-chip Memory Cycles value with improved regex
        CYCLES=$(grep "Off-chip Memory Cycles:" "$LOG_FILE" | grep -o '[0-9]\+' | head -1)
        
        echo "DEBUG: Extracted cycles value: '$CYCLES'"
        
        if [ -n "$CYCLES" ] && [ "$CYCLES" != "" ]; then
            echo "$BS,$CYCLES" >> $CSV_OUTPUT
            echo "  Found cycles for batch size $BS: $CYCLES"
        else
            echo "  Warning: Could not extract cycles from $LOG_FILE"
            # Show the actual line for debugging
            echo "  Debug: $(grep "Off-chip Memory Cycles:" "$LOG_FILE")"
            echo "$BS,N/A" >> $CSV_OUTPUT
        fi
    else
        echo "DEBUG: Log file not found: $LOG_FILE"
        echo "$BS,N/A" >> $CSV_OUTPUT
    fi
    echo "DEBUG: ----------------------------------------"
done

echo ""
echo "CSV file created: $CSV_OUTPUT"
echo "Contents:"
cat $CSV_OUTPUT

# Print usage information
echo ""
echo "Usage: ./validation_output_gather.sh [MEM_CFG] [BS_INCREMENT] [MAX_BS] [PROF_MULTIPLIER]"
echo "Example: ./validation_output_gather.sh spad_off 8 32 1"