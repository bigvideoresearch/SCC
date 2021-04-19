#!/usr/bin/env bash


save_root=../data/google500/myGNN/

# text seed selection config section
threshold_coi=0.7

graph_method=gpu
dist_def=cosine
k=5
self_weight=0
edge_weight=True
train_df_path=../data/google500/data_frame.pkl
train_imglist_path=../data/google500/train_supervised_imglist.txt
train_feature_path=../data/google500/train_supervised_feature.npy
train_score_path=../data/google500/train_supervised_score.npy
train_graph_feature_path=../data/google500/train_supervised_feature.npy

val_imglist_path=../data/google500/test_webvision_supervised_imglist.txt
val_feature_path=../data/google500/test_webvision_supervised_feature.npy
val_graph_feature_path=../data/google500/test_webvision_supervised_feature.npy

cat_ori_feat=True
epochs_num=5000
lr=0.01
weight_decay=1e-6
dropout=0
batch_size=500000
loss_type=bce
reweight=True
train_fast=True


export PYTHONPATH=.

python ./labelers/offline_gnn/train/train_progressive.py \
    --saveroot $save_root \
    --threshold_coi $threshold_coi\
    --train_df_path $train_df_path\
    --graph_method $graph_method \
    --dist_def $dist_def\
    --k $k\
    --self_weight $self_weight\
    --edge_weight $edge_weight\
    --train_imglist_path $train_imglist_path\
    --train_feature_path $train_feature_path\
    --train_score_path $train_score_path\
    --train_graph_feature_path $train_graph_feature_path\
    --val_imglist_path $val_imglist_path\
    --val_feature_path $val_feature_path\
    --val_graph_feature_path $val_graph_feature_path\
    --cat_ori_feat $cat_ori_feat\
    --epochs_num $epochs_num\
    --lr $lr\
    --weight_decay $weight_decay\
    --dropout $dropout\
    --batch_size $batch_size\
    --loss_type $loss_type\
    --reweight $reweight\
    --train_fast $train_fast
