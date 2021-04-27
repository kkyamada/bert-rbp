# BERT-RBP
This repository includes implementations of BERT-RBP, a BERT-based model to predict RNA-RBP interactions. Please cite our paper as well as other dependencies if you use the codes. This repository is still under development, so please report us in case there were any issues.

# 1. Dependencies
For building BERT-RBP:
[DNABERT](https://github.com/jerryji1993/DNABERT)
For analyzing BERT-RBP:
[LinearPartition](https://github.com/LinearFold/LinearPartition)

## 1.1 Install requirements
Install the required packages by running:
```
git clone https://github.com/kkyamada/bert-rbp
cd bert-rbp
python3 -m pip install –editable .
python3 -m pip install -r requirements.txt
```

# 2. Data preprocessing and installation
## 2.1 Data preprocessing
This section is only required if you were to train BERT-RBPs for all 154 RBP data. eCLIP-seq and annotation data for selected RBPs are contained in this repository.

First, download the curated eCLIP-seq data of 154 RBPs from the [RBPsuite](http://www.csbio.sjtu.edu.cn/bioinf/RBPsuite/) website. Then, run the following code to allocate sequences to each RBP and generate datasets for training, evaluation, testing, and analysis (non-training).
```
mkdir datasets
export BENCHMARK_PATH=PATH_TO_BENCHMARK_FILE
export OUTPUT_PATH=./datasets

python3 generate_datasets.py \
	--path_to_benchmark $BENCHMARK_PATH \
	--path_to_output $OUTPUT_PATH
```
You can also utilize --max_num (default 15000), --test_ratio (default 0.2), and --random_seed (default 0) to specify the number of samples to retrieve, the ratio of samples in the test sets, the seed number for random sampling, respectively. 

## 2.2 Download the pre-trained model
Download and unzip the pre-trained DNABERT by following the original instruction [here](https://github.com/jerryji1993/DNABERT). 
If you were conducting the RNA secondary structure analysis (section 4.4 of this page), install LinearPartition by following the original instruction [here](https://github.com/LinearFold/LinearPartition).

# 3. Fine-tuning and evaluation
## 3.1 Download pre-trained DNABERT
Download and unzip the pre-trained DNABERT3 by follwoing the instruction [here](https://github.com/jerryji1993/DNABERT).
If you have skipped 2.1 Data preprocessing, unzip our dataset file by running the following command.
```
tar xzf sample_dataset.tar.gz
```

## 3.2 Fine-tuning
For each RBP, run the following script to train BERT-RBP. The generated model will be saved to the `$OUTPUT_PATH`. Change the name of RBP in `$DATA_PATH` and `$OUTPUT_PATH` as you would like.
```
cd examples

export KMER=3
export MODEL_PATH=PATH_TO_THE_PRETRAINED_MODEL
export DATA_PATH=../datasets/TIAL1/training_sample_finetune
export OUTPUT_PATH=../datasets/TIAL1/finetuned_model

python3 run_finetune_bertrbp.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 101 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8 \
    --num_gpu 4 \
    --num_node 1
```
Then, run the following script to compute the prediction performance of each BERT-RBP. Scores will be recorded as `eval_results_prediction.txt` file in the `$MODEL_PATH`. 
```
export KMER=3
export MODEL_PATH=../datasets/TIAL1/finetuned_model
export DATA_PATH=../datasets/TIAL1/test_sample_finetune

python3 run_finetune_bertrbp.py \
	--model_type dna \
	--tokenizer_name dna$KMER \
	--model_name_or_path $MODEL_PATH \
	--task_name dnaprom \
	--do_eval \
	--do_predict \
	--data_dir $DATA_PATH \
	--output_dir $MODEL_PATH \
	--predict_dir $MODEL_PATH \
	--max_seq_length 101 \
	--per_gpu_eval_batch_size 32 \
	--per_gpu_train_batch_size 32 \
	--overwrite_output \
	--num_gpu 4 \
	--num_node 1
```

# 4. Attention analysis
## 4.1 Region type annotation
For TIAL1 and EWSR1, annotation files are contained in the `sample_dataset/RBP/nontraining_sample_finetune` as `annotation.npy`. When creating annotation files of your interest, refer to the above file format or see the instruction in the `annotations.ipynb` file.

## 4.2 Region type analysis
After fine-tuning, you can conduct region type analysis on the specified BERT-RBP  by running: 
```
export RBP=TIAL1
export MODEL_PATH=../datasets/TIAL1/finetuned_model
export DATA_PATH=../datasets/TIAL1/nontraining_sample_finetune
export PRED_PATH=../datasets/TIAL1/finetuned_model/analyze_regiontype 
python3 run_finetune_bertrbp.py \
--model_type dna \
--tokenizer_name dna3 \
--model_name_or_path $MODEL_PATH \
--task_name dnaprom \
--do_analyze_regiontype \
--visualize_data_dir $DATA_PATH \
--data_dir $DATA_PATH \
--max_seq_length 101 \
--per_gpu_pred_batch_size 128 \
--output_dir $MODEL_PATH \
--predict_dir $PRED_PATH \
--n_process 8 \
--num_gpu 1 \
--num_node 1 \
--region_type 0 \
--rbp_name $RBP
```
The results of analysis will be exported to the `$PRED_PATH`. To visualize the results, follow the instruction in the `visualization.ipynb` file.

For detailed analysis, run the following command:
```
export RBP=TIAL1
export MODEL_PATH=../datasets/TIAL1/finetuned_model
export DATA_PATH=../datasets/TIAL1/nontraining_sample_finetune
export PRED_PATH=../datasets/TIAL1/finetuned_model/analyze_regiontype 
export SPECIFIC_HEADS="(1,9,1),(2,9,11),(4,12,4),(5,12,4)"

python3 run_finetune_bertrbp.py \
--model_type dna \
--tokenizer_name dna3 \
--model_name_or_path $MODEL_PATH \
--task_name dnaprom \
--do_analyze_regiontype_specific \
--visualize_data_dir $DATA_PATH \
--data_dir $DATA_PATH \
--max_seq_length 101 \
--per_gpu_pred_batch_size 128 \
--output_dir $MODEL_PATH \
--predict_dir $PRED_PATH \
--n_process 8 \
--num_gpu 1 \
--num_node 1 \
--specific_heads $SPECIFIC_HEADS
```
`$SPECIFIC_HEADS` indicates the heads where attention ratio were the highest for each region type (1=5’UTR, 2=3’UTR, 3=exon, 4=intron, 5=CDS). For instance, `(2,9,11)` indicates that the 11th head in the 9th layer showed the highest attention ratio for 3’UTR.

## 4.3 Region boundary analysis
Region boundary analysis for each BERT-RBP can be conducted by running:
```
export MODEL_PATH=../datasets/TIAL1/finetuned_model
export DATA_PATH=../datasets/TIAL1/nontraining_sample_finetune
export PRED_PATH=../datasets/TIAL1/finetuned_model/analyze_regionboundary 

python3 run_finetune_bertrbp.py \
--model_type dna \
--tokenizer_name dna3 \
--model_name_or_path $MODEL_PATH \
--task_name dnaprom \
--do_analyze_regionboundary \
--visualize_data_dir $DATA_PATH \
--data_dir $DATA_PATH \
--max_seq_length 101 \
--per_gpu_pred_batch_size 128 \
--output_dir $MODEL_PATH \
--predict_dir $PRED_PATH \
--n_process 8 \
--num_gpu 1 \
--num_node 1  \
--rbp_name $RBP
```
The results of analysis will be exported to the `$PRED_PATH`. To visualize the results, follow the instruction in the `visualization.ipynb` file. When you further conduct detailed analysis, replace the `--do_analyze_regionboundary` to `--do_analyze_regionboundary_specific` and specify heads with `--specific_heads $SPECIFIC_HEADS`. Note that `export SPECIFIC_HEADS=”(1,2),(3,4)”` (the 2nd head in the 1st layer and 4th head in the 3rd layer, in this example) should be defined at the beginning. 

## 4.4 Secondary structure analysis
Note that you need to install LinearPartition before this section. RNA secondary structure analysis for each BERT-RBP can be conducted by running:
```
export MODEL_PATH=../datasets/EWSR1/finetuned_model
export DATA_PATH=../datasets/EWSR1/nontraining_sample_finetune
export PRED_PATH=../datasets/EWSR1/finetuned_model/analyze_rnastructure
export LINEARPARTITION_PATH=PATH_TO_LINEARPARTITION

python3 run_finetune_bertrbp.py \
--model_type dna \
--tokenizer_name dna3 \
--model_name_or_path $MODEL_PATH \
--task_name dnaprom \
--do_analyze_rnastructure \
--visualize_data_dir $DATA_PATH \
--data_dir $DATA_PATH \
--max_seq_length 101 \
--per_gpu_pred_batch_size 128 \
--output_dir $MODEL_PATH \
--predict_dir $PRED_PATH \
--path_to_linearpartition $LINEARPARTITION_PATH \
--n_process 8 \
--num_gpu 1 \
--num_node 1 \
--rbp_name $RBP
```
The results of analysis will be exported to the `$PRED_PATH`. To visualize the results, follow the instruction in the `visualization.ipynb` file. When you further conduct detailed analysis, replace the `--do_analyze_rnastructure` to `--do_analyze_rnastructure_specific` and specify heads with `--specific_heads $SPECIFIC_HEADS`. Note that `export SPECIFIC_HEADS=”(1,2),(3,4)”` (the 2nd head in the 1st layer and 4th head in the 3rd layer, in this example) should be defined at the beginning.

# 5. Citations
If you used BERT-RBP in your research, please kindly cite the following paper.

```
to be updated
```

Also, the following papers are major prior works on which our research is based.
```
@ARTICLE{Ji2021-ie,
  title    = "{DNABERT}: pre-trained Bidirectional Encoder Representations from Transformers model for {DNA-language} in genome",
  author   = "Ji, Yanrong and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V",
  journal  = "Bioinformatics",
  pages    = "2020.09.17.301879",
  month    =  feb,
  year     =  2021,
  language = "en",
  url     = "http://dx.doi.org/10.1093/bioinformatics/btab083"
}
```
```
@ARTICLE{Pan2020-he,
  title     = "{RBPsuite}: {RNA-protein} binding sites prediction suite based
               on deep learning",
  author    = "Pan, Xiaoyong and Fang, Yi and Li, Xianfeng and Yang, Yang and Shen, Hong-Bin",
  journal   = "BMC genomics",
  volume    =  21,
  number    =  1,
  pages     = "884",
  month     =  dec,
  year      =  2020,
  language  = "en",
  url    = "http://dx.doi.org/10.1186/s12864-020-07291-6"
}
```

![image](https://user-images.githubusercontent.com/70200890/116222943-00762400-a78a-11eb-89aa-53617cd21835.png)
