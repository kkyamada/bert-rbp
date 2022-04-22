#!/bin/bash
ORIG_PATH=../sample_dataset/

# RBP=$1
# echo $RBP
RBP=TIAL1

PYTHON_PATH=../examples/run_analysis_motif.py
MODEL_PATH=$ORIG_PATH$RBP/finetuned_model
DATA_PATH=$ORIG_PATH$RBP/test_sample_finetune/
PREDICT_PATH=$MODEL_PATH/atten_data

python3 $PYTHON_PATH --model_type dna --model_name_or_path $MODEL_PATH --task_name dnaprom --data_dir $DATA_PATH --predict_dir $PREDICT_PATH --overwrite_output_dir --do_analyze_motif --max_seq_length 101 --per_gpu_pred_batch_size 64 --n_process 8

PYTHON_PATH=../motif/find_motifs.py
MOTIF_PATH=$PREDICT_PATH/motif_saved/

python3 $PYTHON_PATH --data_dir $DATA_PATH --predict_dir $PREDICT_PATH --window_size 12 --min_len 5 --max_len 10 --pval_cutoff 0.005 --min_n_motif 2 --top_n_motif 10 --align_all_ties --save_file_dir $MOTIF_PATH --verbose --kmer $KMER

