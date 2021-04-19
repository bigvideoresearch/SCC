#dataset=food101n/food101n
dataset=food101n/food101n


mixup_alpha=0.2
pipeline_setting=mixup${mixup_alpha}


python -W ignore -u main.py \
-cfg configs/pipelines/baseline/baseline_res50v1d_learn.yaml \
configs/pipelines/baseline/reweight_learn.yaml \
configs/datasets/${dataset}.yaml \
configs/pipelines/baseline/mixup_learn.yaml \
--data.train.transform_batch.Mixup.alpha ${mixup_alpha} \
--pipeline_setting ${pipeline_setting} \
2>&1 | tee -a ../log.txt \
