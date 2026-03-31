python flowlet/train.py \
        experiment=flowlet_openbhb_paper model=flowlet_large \
        data.data_dir=/root/DATASETS/openbhb_data/train/ \
        data.metadata_file=/root/DATASETS/openbhb_data/train.tsv \
        logger=wandb
