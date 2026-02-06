#!/bin/bash
# Quick test script for OpenBHB dataset

echo "============================================================"
echo "Quick OpenBHB Dataset Test"
echo "============================================================"
echo

# Test 1: Check files exist
echo "1. Checking files..."
if [ -f "/home/fred/Projetos/Einstein/openbhb_train_sample/train.tsv" ]; then
    echo "   ✓ Metadata file exists"
else
    echo "   ✗ Metadata file NOT found"
    exit 1
fi

if [ -d "/home/fred/Projetos/Einstein/openbhb_train_sample/train/quasiraw_3d" ]; then
    echo "   ✓ Data directory exists"
    NUM_FILES=$(ls /home/fred/Projetos/Einstein/openbhb_train_sample/train/quasiraw_3d/*.npy 2>/dev/null | wc -l)
    echo "   ✓ Found $NUM_FILES .npy files"
else
    echo "   ✗ Data directory NOT found"
    exit 1
fi

# Test 2: Load dataset with Python
echo
echo "2. Testing dataset loading..."
python3 << 'TESTEOF'
import sys
sys.path.insert(0, '/home/fred/Projetos/Einstein/FlowLet/flowlet/data')

try:
    from openbhb_dataset import OpenBHBDataset

    dataset = OpenBHBDataset(
        data_dir="/home/fred/Projetos/Einstein/openbhb_train_sample",
        metadata_file="/home/fred/Projetos/Einstein/openbhb_train_sample/train.tsv",
        target_shape=(128, 128, 128),
        normalize=True,
    )

    stats = dataset.get_age_statistics()

    print(f"   ✓ Dataset loaded: {stats['count']} samples")
    print(f"   ✓ Age range: {stats['min']:.1f} - {stats['max']:.1f} years")

    # Load one sample
    sample = dataset[0]
    print(f"   ✓ Sample loaded: {sample['volume'].shape}")

    print("\n✅ All tests passed!")
    exit(0)

except Exception as e:
    print(f"\n❌ Error: {e}")
    exit(1)
TESTEOF

TEST_RESULT=$?

echo
echo "============================================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ OpenBHB dataset is ready for training!"
    echo "============================================================"
    echo
    echo "Next step: Start fine-tuning"
    echo
    echo "Command:"
    echo "  python flowlet/train.py \\"
    echo "    experiment=flowlet_openbhb_finetune \\"
    echo "    ckpt_path=/path/to/fomo60k_checkpoint.ckpt"
else
    echo "❌ Tests failed. Check errors above."
    echo "============================================================"
fi
