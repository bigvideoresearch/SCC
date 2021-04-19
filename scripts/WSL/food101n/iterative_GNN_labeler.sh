#!/bin/bash
total_epoch=120
start_epoch=120
recovery_epoch=60
num_gpus=8
dataset=food101n/food101n


saveroot=../data/Google500/
save_cnn_filename=Google500_cnn_pseudo_epoch
save_gnn_filename=Google500_gnn_pseudo_epoch
save_merge_filename=../data/google500/myGNN_th_0.7_lambda_0.5_score/
load_model_path=../checkpoint/best_valid_web_top1.pth
lr=0.05
conf_policy=Distillation
################
model_path=tbd #
################

# initialized by training start_epoch from scratch

python -W ignore -u main.py \
-cfg configs/pipelines/baseline/baseline_res50v1d_learn.yaml \
configs/pipelines/baseline/reweight_learn.yaml \
configs/datasets/${dataset}.yaml \
--num_epochs ${start_epoch} \
2>&1 | tee -a ../log.txt

# every recovery_epoch, a repeatable pipeline is listed:
#for i in $(seq ${start_epoch} ${recovery_epoch} ${total_epoch}); do
	# extract features

python -W ignore -u main.py \
-cfg configs/pipelines/baseline/baseline_res50v1d_learn.yaml \
configs/datasets/${dataset}.yaml \
configs/pipelines/tools/feature_extractor.yaml \
--save_root ${saveroot} \
--save_filename ${save_cnn_filename} \
--num_epochs ${start_epoch} \
2>&1 | tee -a ../log.txt

# get gnn labels
sh ./labelers/offline_gnn/sgc_labeler.sh ${save_cnn_filename} ${save_gnn_filename}

	# merge cnn labels and gnn labels
python ./labelers/offline_gnn/mergeScore.py --threshold 0.7 \
--file_root '../data/google500/' \
--sgc_index 'myGNN'

# distillation with merged cnn ahd gnn labels

python -W ignore -u main.py \
-cfg configs/pipelines/baseline/baseline_res50v1d_learn.yaml \
configs/pipelines/baseline/reweight_learn.yaml \
configs/datasets/${dataset}.yaml \
configs/pipelines/WSL/offline_SCC_learn.yaml \
--data.train.pseudo_root ${save_merge_filename} \
--num_gpus ${num_gpus} \
--num_epochs ${total_epoch} \
--load_model_path ${load_model_path} \
--lr ${lr} \
--data.train.imglist_path ${saveroot}${save_cnn_filename}_imglist.txt \
--SCC_setting.conf_policy ${conf_policy} \
2>&1 | tee -a ../log.txt
