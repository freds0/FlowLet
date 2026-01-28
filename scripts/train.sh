#!/bin/bash
# FlowLet Training Script
#
# Usage:
#   ./scripts/train.sh                    # Train with synthetic data (for testing)
#   ./scripts/train.sh oasis             # Train on OASIS dataset
#   ./scripts/train.sh fomo60k           # Train on FOMO60k dataset
#   ./scripts/train.sh fomo60k_min       # Train on FOMO60k with low memory
#   ./scripts/train.sh min_memory        # Train with minimal memory footprint
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES=0,1  # GPU selection
#   WANDB_MODE=offline        # Disable W&B logging

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Default experiment
EXPERIMENT="${1:-flowlet_synthetic}"

echo "=================================================="
echo "FlowLet Training"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Experiment: $EXPERIMENT"
echo "=================================================="

# Map short names to full experiment names
case "$EXPERIMENT" in
    synthetic|test)
        EXPERIMENT="flowlet_synthetic"
        ;;
    oasis)
        EXPERIMENT="flowlet_oasis"
        ;;
    fomo|fomo60k)
        EXPERIMENT="flowlet_fomo60k"
        ;;
    fomo_min|fomo60k_min)
        EXPERIMENT="flowlet_fomo60k_min_memory"
        ;;
    min_memory|low_mem|small)
        EXPERIMENT="flowlet_min_memory"
        ;;
esac

echo "Running experiment: $EXPERIMENT"
echo ""

# Run training
python flowlet/train.py experiment="$EXPERIMENT" "${@:2}"

echo ""
echo "=================================================="
echo "Training complete!"
echo "=================================================="
