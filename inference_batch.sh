#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default Configurations
N_SAMPLES=7                    # Total number of samples (N) to generate
CHECKPOINT_DIR=""              # Folder containing the .ckpt file (Required input)
N_STEPS=10                     # Number of ODE steps per synthesis
OUTPUT_DIR="./synthesized_batch" # Final directory to store the images

# Help / Usage function
show_help() {
    echo "Usage: $0 --checkpoint_dir <path_to_folder> [OPTIONS]"
    echo ""
    echo "Required Options:"
    echo "  -c, --checkpoint_dir PATH   Path to the folder containing the .ckpt checkpoint file"
    echo ""
    echo "Optional Options:"
    echo "  -n, --n_samples INT         Total number N of images to generate between ages 20 and 80 (Default: 7)"
    echo "  -s, --n_steps INT           Number of ODE sampling steps (Default: 10)"
    echo "  -o, --output_dir PATH       Directory where images will be saved (Default: ./synthesized_batch)"
    echo "  -h, --help                  Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        -n|--n_samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        -s|--n_steps)
            N_STEPS="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Error: Unknown parameter $1"
            show_help
            ;;
    esac
done

# Validate required parameter
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: The --checkpoint_dir parameter is required."
    show_help
fi

echo "========================================================================"
echo "          STARTING BATCH SYNTHESIS - FLOWLET (20 TO 80 YEARS)           "
echo "========================================================================"
echo "-> Checkpoint Directory: $CHECKPOINT_DIR"
echo "-> Total Samples (N):    $N_SAMPLES"
echo "-> Steps per Sample:     $N_STEPS"
echo "-> Output Directory:     $OUTPUT_DIR"
echo "------------------------------------------------------------------------"

# Loop to calculate and generate each sample conditioned on age
for (( i=0; i<N_SAMPLES; i++ )); do
    # If N is 1, just generate the initial age (20) to avoid division by zero
    if [ "$N_SAMPLES" -eq 1 ]; then
        age=20
    else
        # Linearly distribute the age between 20 and 80 using bc arithmetic
        # Formula: age = 20 + i * (80 - 20) / (N_SAMPLES - 1)
        #age=$(echo "scale=2; 20 + $i * (80 - 20) / ($N_SAMPLES - 1)" | bc)
        age=$(echo "scale=2; 50 + $i * (80 - 20) / ($N_SAMPLES - 1)" | bc)
    fi

    echo ""
    echo "[Sample $((i+1))/$N_SAMPLES] Running inference for AGE: ${age} years..."
    
    # Call the previously configured Python inference script
    python inference.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --age "$age" \
        --n_steps "$N_STEPS" \
        --output_dir "$OUTPUT_DIR"

done

echo ""
echo "========================================================================"
echo " ✓ Process completed successfully! All $N_SAMPLES images are saved in: $OUTPUT_DIR"
echo "========================================================================"
