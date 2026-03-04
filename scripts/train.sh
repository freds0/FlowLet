#!/bin/bash
# FlowLet Training Script
#
# Usage:
#   ./scripts/train.sh                          # Train with paper config (OpenBHB)
#   ./scripts/train.sh flowlet_openbhb_paper    # Explicit experiment name
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
EXPERIMENT="${1:-flowlet_openbhb_paper}"

echo "=================================================="
echo "FlowLet Training"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Experiment: $EXPERIMENT"
echo "=================================================="

echo "Running experiment: $EXPERIMENT"
echo ""

# Run training
python flowlet/train.py experiment="$EXPERIMENT" "${@:2}"

echo ""
echo "=================================================="
echo "Training complete!"
echo "=================================================="
