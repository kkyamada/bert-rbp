# BERT-RBP
This repository includes implementations of BERT-RBP, a BERT-based model to predict RNA-RBP interactions. Please cite our paper as well as other dependencies if you use the codes. This repository is still under development, so please report to us in case there were any issues.

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
python3 -m pip install –-editable .
python3 -m pip install -r requirements_bertrbp.txt
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
Download and unzip the pre-trained DNABERT3 by following the instruction [here](https://github.com/jerryji1993/DNABERT).
If you have skipped 2.1 Data preprocessing, unzip our dataset file by running the following command.
```
tar xzf sample_dataset.tar.gz
```

## 3.2 Fine-tuning
For each RBP, run the following script to train BERT-RBP. The generated model will be saved to the `$OUTPUT_PATH`. Change the name of RBP in `$DATA_PATH` and `$OUTPUT_PATH` as you would like. Use the additional argument, `--do_train_from_scratch`, to train BERT-baseline, whose model parameters were randomly initialized.  
```
cd examples

export KMER=3
export MODEL_PATH=PATH_TO_THE_PRETRAINED_DNABERT
export DATA_PATH=../sample_dataset/TIAL1/training_sample_finetune
export OUTPUT_PATH=../sample_dataset/TIAL1/finetuned_model

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
## 3.3 Measure prediction performance
Run the following script to compute the prediction performance of each BERT-RBP. Scores will be recorded as `eval_results_prediction.txt` file in the `$MODEL_PATH`. 
```
export KMER=3
export MODEL_PATH=../sample_dataset/TIAL1/finetuned_model
export DATA_PATH=../sample_dataset/TIAL1/test_sample_finetune

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
For TIAL1 and EWSR1, annotation files are contained in the `sample_dataset/RBP/nontraining_sample_finetune` as `annotations.npy`. When creating annotation files of your interest, refer to the above file format or see the instruction in the `annotations.ipynb` file.

## 4.2 Region type analysis
After fine-tuning, you can conduct region type analysis on the specified BERT-RBP  by running: 
```
export RBP=TIAL1
export MODEL_PATH=../sample_dataset/TIAL1/finetuned_model
export DATA_PATH=../sample_dataset/TIAL1/nontraining_sample_finetune
export PRED_PATH=../sample_dataset/TIAL1/finetuned_model/analyze_regiontype 

python3 run_finetune_bertrbp.py \
	--model_type dna \
	--tokenizer_name dna3 \
	--model_name_or_path $MODEL_PATH \
	--task_name dnaprom \
	--do_analyze_regiontype \
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
The results of the analysis will be exported to the `$PRED_PATH`. To visualize the results, follow the instruction in the `visualization.ipynb` file. For detailed analysis, replace the `--do_analyze_regiontype` to `--do_analyze_regiontype_specific`.

## 4.3 Region boundary analysis
Region boundary analysis for each BERT-RBP can be conducted by running:
```
export MODEL_PATH=../sample_dataset/EWSR1/finetuned_model
export DATA_PATH=../sample_dataset/EWSR1/nontraining_sample_finetune
export PRED_PATH=../sample_dataset/EWSR1/finetuned_model/analyze_regionboundary 

python3 run_finetune_bertrbp.py \
	--model_type dna \
	--tokenizer_name dna3 \
	--model_name_or_path $MODEL_PATH \
	--task_name dnaprom \
	--do_analyze_regionboundary \
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
The results of the analysis will be exported to the `$PRED_PATH`. To visualize the results, follow the instruction in the `visualization.ipynb` file. When you further conduct detailed analysis, replace the `--do_analyze_regionboundary` to `--do_analyze_regionboundary_specific`.

## 4.4 Secondary structure analysis
Note that you need to install LinearPartition before this section. RNA secondary structure analysis for each BERT-RBP can be conducted by running:
```
export MODEL_PATH=../datasets/HNRNPK/finetuned_model
export DATA_PATH=../datasets/HNRNPK/nontraining_sample_finetune
export PRED_PATH=../datasets/HNRNPK/finetuned_model/analyze_rnastructure
export LINEARPARTITION_PATH=PATH_TO_LINEARPARTITION

python3 run_finetune_bertrbp.py \
	--model_type dna \
	--tokenizer_name dna3 \
	--model_name_or_path $MODEL_PATH \
	--task_name dnaprom \
	--do_analyze_rnastructure \
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
The results of the analysis will be exported to the `$PRED_PATH`. To visualize the results, follow the instruction in the `visualization.ipynb` file. When you further conduct detailed analysis, replace the `--do_analyze_rnastructure` with `--do_analyze_rnastructure_specific`.

# 5. Citations
If you used BERT-RBP in your research, please cite the following paper.

```
@UNPUBLISHED{Yamada2021-ql,
  title    = "Prediction of {RNA-protein} interactions using a nucleotide
              language model",
  author   = "Yamada, Keisuke and Hamada, Michiaki",
  journal  = "bioRxiv",
  pages    = "2021.04.27.441365",
  month    =  apr,
  year     =  2021,
  language = "en"
}
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
