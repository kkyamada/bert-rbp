#!/bin/bash

PYTHON_PATH=../examples/create_secondary_structure.py
ORIG_PATH=../sample_dataset/
PATH2=nontraining_sample_finetune/hg38/
NUM_EACH=1000

RBP=$1
echo $RBP
# RBP=TIAL1
PATH_TO_LINEARPARTITION=$2
echo $PATH_TO_LINEARPARTITION

python3 $PYTHON_PATH --rbp $RBP --data_dir $ORIG_PATH --path_to_linearpartition $PATH_TO_LINEARPARTITION --data_dir_suffix $PATH2 --num_each $NUM_EACH

echo "done"
