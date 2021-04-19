dataset=food101n/food101n

pipeline_setting=baseline_TEST


python -W ignore -u main.py \
-cfg configs/pipelines/baseline/baseline_res50v1d_learn.yaml \
configs/pipelines/baseline/reweight_learn.yaml \
configs/datasets/${dataset}.yaml \
--pipeline_setting ${pipeline_setting} \
2>&1 | tee -a ../log.txt
