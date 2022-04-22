#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l gpu
#$ -l cuda=2
# 128G => n_process=24
#$ -l s_vmem=32G 
#$ -l mem_req=32G
#$ -l d_rt=8:00:00
#$ -l s_rt=8:00:00
#$ -e ./log
#$ -o ./log
#$ -V

module load singularity
module load cuda10.0/toolkit/10.0.130

DOCKER_PATH=/home/keisuke-yamada/dockerimages/docker_dnabert_latest.sif

PYTHON_PATH=../examples/run_analysis_structure.py
ORIG_PATH=../sample_dataset/

# RBP=$1
# echo "RBP: "$RBP
RBP=TIAL1

MODEL_PATH=$ORIG_PATH$RBP/finetuned_model

DATA_PATH=$ORIG_PATH$RBP/nontraining_sample_finetune/hg38/

PREDICT_PATH=$MODEL_PATH/analyze_rnastructure/

singularity exec --nv -e $DOCKER_PATH python3 $PYTHON_PATH --model_type dna --model_name_or_path $MODEL_PATH --task_name dnaprom --data_dir $DATA_PATH --predict_dir $PREDICT_PATH --do_analyze_rnastructure --max_seq_length 101 --per_gpu_pred_batch_size 64 --n_process 8

singularity exec --nv -e $DOCKER_PATH python3 $PYTHON_PATH --model_type dna --model_name_or_path $MODEL_PATH --task_name dnaprom --data_dir $DATA_PATH --predict_dir $PREDICT_PATH --do_analyze_rnastructure_specific --max_seq_length 101 --per_gpu_pred_batch_size 64 --n_process 8

echo "done"