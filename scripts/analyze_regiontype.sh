#!/bin/bash

PYTHON_PATH=../examples/run_analysis_regiontype.py
ORIG_PATH=../sample_dataset/

# RBP=$1
# echo "RBP: "$RBP
RBP=TIAL1

MODEL_PATH=$ORIG_PATH$RBP/finetuned_model

DATA_PATH=$ORIG_PATH$RBP/nontraining_sample_finetune/hg38/

PREDICT_PATH=$MODEL_PATH/analyze_regiontype/

python3 $PYTHON_PATH --model_type dna --model_name_or_path $MODEL_PATH --task_name dnaprom --data_dir $DATA_PATH --predict_dir $PREDICT_PATH --do_analyze_regiontype --max_seq_length 101 --per_gpu_pred_batch_size 64 --n_process 8

echo "done"