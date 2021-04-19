#!/bin/bash
num_gpus=1
total_epoch=120
# # Important: usually be the half of original lr
lr=0.05
dataset=food101n/food101n
mixup_alpha=0.2
pipeline_setting=mixup${mixup_alpha}
# confidence usage policy
# select from (ConfvsOneMinusConf, ConstantConf_0.5, Distillation)
conf_policy=ConfvsOneMinusConf

# model path to use
load_model_path=../checkpoint/best_valid_avg_top1.pth
# Saving Features and Pseudo Labels
saveroot=../data/
save_cnn_filename=food101n_0128

# extract features and scores


python -W ignore -u main.py \
-cfg configs/pipelines/baseline/baseline_res50v1d_learn.yaml \
configs/datasets/${dataset}.yaml \
configs/pipelines/tools/feature_extractor.yaml \
--num_gpus ${num_gpus} \
--data.num_workers 0 \
--load_model_path ${load_model_path} \
--save_root ${saveroot} \
--save_filename ${save_cnn_filename} \
2>&1 | tee -a ../log.txt

# merge cnn labels and gnn labels
python ./labelers/offline_cnn/distribute_score_file.py --file_root ${saveroot} \
--keyword ${save_cnn_filename} --get_new_reweight False

# Offline SCC

python -W ignore -u main.py \
-cfg configs/pipelines/baseline/baseline_res50v1d_learn.yaml \
configs/pipelines/baseline/reweight_learn.yaml \
configs/datasets/${dataset}.yaml \
configs/pipelines/baseline/mixup_learn.yaml \
configs/pipelines/WSL/offline_SCC_learn.yaml \
--data.train.pseudo_root ${saveroot}${save_cnn_filename}_score \
--num_gpus ${num_gpus} \
--num_epochs ${total_epoch} \
--load_model_path ${load_model_path} \
--lr ${lr} \
--data.train.imglist_path ${saveroot}${save_cnn_filename}_imglist.txt \
--data.train.transform_batch.Mixup.alpha ${mixup_alpha} \
--pipeline_setting ${pipeline_setting} \
--SCC_setting.conf_policy ${conf_policy} \
2>&1 | tee -a ../log.txt
