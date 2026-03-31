python flowlet/train.py \
        experiment=flowlet_openbhb_paper model=flowlet_large \
        data.data_dir=/home/fred/Projetos/Einstein/OpenBHB_Dataset/openbhb_train_sample/train \
        data.metadata_file=/home/fred/Projetos/Einstein/OpenBHB_Dataset/openbhb_train_sample/train.tsv \
        logger=tensorboard
