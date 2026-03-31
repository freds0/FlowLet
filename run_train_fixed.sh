#!/bin/bash
python flowlet/train.py \
    experiment=flowlet_openbhb_paper model=flowlet_large \
    data.data_dir=/root/DATASETS/openbhb_data/train/ \
    data.metadata_file=/root/DATASETS/openbhb_data/train.tsv \
    data.num_workers=0 \
    +trainer.num_sanity_val_steps=0 \
    logger=wandb

#python flowlet/train.py experiment=flowlet_openbhb_cached model=flowlet_large logger=wandb

