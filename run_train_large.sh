#python flowlet/train.py \
#        experiment=flowlet_openbhb_paper model=flowlet_large \
#        data.data_dir=/home/fred/Projetos/Einstein/OpenBHB_Dataset/openbhb_train_sample/train \
#        data.metadata_file=/home/fred/Projetos/Einstein/OpenBHB_Dataset/openbhb_train_sample/train.tsv \
#        logger=tensorboard


python flowlet/train.py experiment=flowlet_fomo300k \
  model=flowlet_large \
#  logger=wandb \
#  logger.wandb.project=flowlet \
#  logger.wandb.name=fomo300k-train
