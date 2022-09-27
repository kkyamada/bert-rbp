# BERT-RBP
This repository includes implementations of BERT-RBP, a BERT-based model to predict RNA-RBP interactions. Please cite our paper as well as other dependencies if you use the codes. This repository is still under development, so please report to us in case there were any issues.

# 1. Dependencies
For building BERT-RBP:
[DNABERT](https://github.com/jerryji1993/DNABERT)
For analyzing BERT-RBP:
[LinearPartition](https://github.com/LinearFold/LinearPartition)
We have tested our program in the following environments.
```
Linux: x86_64
GPU: NVIDIA Tesla V100 SXM2
CUDA Version: 10.0
Nvidia Driver Version: 440.33.01
```
and
```
Linux: x86_64
GPU: NVIDIA Tesla V100 DGXS
CUDA Version: 10.0
Nvidia Driver Version: 440.33.01
```

## 1.1 Install requirements
Install the required packages by running:
```
git clone https://github.com/kkyamada/bert-rbp
cd bert-rbp
python3 -m pip install -r requirements_bertrbp.txt
```
or if you use pipenv, by runnning:
```
git clone https://github.com/kkyamada/bert-rbp
cd bert-rbp
pipenv sync
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
If you have skipped 2.1 Data preprocessing, use the sample TIAL1 data in `sample_dataset`.

## 3.2 Fine-tuning and prediction
For each RBP, use `scripts/train_and_test.sh` by running the following command to train BERT-RBP.
```
cd scripts
source trian_and_test.sh TIAL1 PATH_YOU_SAVED_DNABERT
```
The generated model will be saved to the `$OUTPUT_PATH`. Modify the bash file to change the name of RBP in `$DATA_PATH` and `$OUTPUT_PATH` as you would like. Use the additional argument, `--do_train_from_scratch`, to train BERT-baseline, whose model parameters will be randomly initialized.  
In our setting, we trained the model using 4 GPUs with the batch size of 32. When you train the model, make sure the total batch size (number of GPU * batch size per GPU * gradient acculumation steps) is >= 128.  
If you were to conduct prediction on sequences whose labels are unknown, first create `dev.tsv` file in the same format as those in the sample dataset. (Set their labels (0 or 1) arbitarily as they were not used during prediction anyway.) Then, run the following command, and it will generate the prediction in `pred_results.npy`. This .npy file will contain a vector with the size of the number of sequences, where each element represents a label1 score for each sequence. The predicted label is 1 when this score is >0.5, otherwise the prediction is 0.
```
cd scripts
source prediction.sh TIAL1
```

# 4. Attention analysis
## 4.1 Region type annotation
For TIAL1, annotation file is contained in the `sample_dataset/TIAL1/nontraining_sample_finetune/hg38` as `annotations_binary.npy`. When creating annotation files of your interest, refer to the above file format or see the instruction in the `annotations.ipynb` file.

## 4.2 Region type analysis
After fine-tuning, you can conduct region type analysis on the specified BERT-RBP by using `scripts/analyze_regiontype.sh`.
```
source analyze_regiontype.sh TIAL1
```
The results of the analysis will be exported to the `./sample_dataset/TIAL1/finetuned_model/analyze_reigontype/` , which is defined as `$PRED_PATH` in the script. To visualize the results, follow the instruction in the `visualization.ipynb` file.

## 4.3 Secondary structure analysis
Note that you need to install LinearPartition before this section. First, generate RNA secondary structure labels by running `scripts/generate_2dstructure.sh`. RNA secondary structure analysis for each BERT-RBP can be conducted by using `scripts/analyze_2dstructure.sh`.

```
source generate_2dstructure.sh TIAL1 PATH_YOU_SAVED_LINEARPARTITION
source analyze_2dstructure.sh TIAL1
```
The results of the analysis will be exported to the `./sample_dataset/TIAL1/finetuned_model/analyze_rnastructure/` , which is defined as `$PRED_PATH`. To visualize the results, follow the instruction in the `visualization.ipynb` file.

# 5. Citations
If you used BERT-RBP in your research, please cite the following paper.

```
@article{10.1093/bioadv/vbac023,
    author = {Yamada, Keisuke and Hamada, Michiaki},
    title = "{Prediction of RNA-protein interactions using a nucleotide language model}",
    journal = {Bioinformatics Advances},
    year = {2022},
    month = {04},
    issn = {2635-0041},
    doi = {10.1093/bioadv/vbac023},
    url = {https://doi.org/10.1093/bioadv/vbac023},
    note = {vbac023},
    eprint = {https://academic.oup.com/bioinformaticsadvances/advance-article-pdf/doi/10.1093/bioadv/vbac023/43296569/vbac023.pdf},
}
```
