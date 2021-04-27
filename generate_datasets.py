import sys
sys.path.append('./motifs')
from motif_utils import seq2kmer
import argparse
import numpy as np
import pandas as pd
import re
import os

OUTPUT_FILE = 'original.tsv'

def createkmers(args):
    with open(args.path_to_benchmark) as f:
        lines = f.readlines()
    
    filename = 'EMPTY'
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'train_dir' in line:
            pattern = '([A-Z0-9]+.negative)|([A-Z0-9]+.positive)'
            filename = re.sub('[.]', '_', re.search(pattern, line).group())
            # print(filename)
            value = 0
            if 'positive' in filename:
                value = 1
            filename = re.search('[A-Z0-9]+', filename).group()
            filepath = os.path.join(args.path_to_output, filename, OUTPUT_FILE)
            
            if os.path.exists(filepath):
                with open(filepath, mode='a') as f:
                    i += 1
                    flg = 0
                    while flg == 0:
                        if not '>' in lines[i]:
                            f.write(seq2kmer(re.sub('U', 'T', lines[i][:-1]), args.kmer) + '\t' +str(value) + '\n')
                        i += 1
                        if i >= len(lines):
                            break
                        elif 'train_dir' in lines[i]:
                            break
            else:
                if os.path.exists(args.path_to_output):
                    os.mkdir(args.path_to_output + '/' + filename)
                else:
                    os.mkdir(args.path_to_output)
                    os.mkdir(args.path_to_output + '/' + filename)
                with open(filepath, mode='w') as f:
                    f.write('sequence' + '\t' + 'label\n')
                    i += 1
                    flg = 0
                    while flg == 0:
                        if not '>' in lines[i]:
                            f.write(seq2kmer(re.sub('U', 'T', lines[i][:-1]), args.kmer) + '\t' + str(value) + '\n')
                        i += 1
                        if i >= len(lines):
                            break
                        elif 'train_dir' in lines[i]:
                            break
        else:
            print('ERROR : ', args.path_to_output)
    return

def preprocess(args):
    orig_dir = os.path.join(args.path_to_output)
    dirlist = os.listdir(orig_dir)
    dirlist.sort()
    # print(dirlist)

    len_sequence_list = []
    for rbp in dirlist:
        one_sequence_list = []
        rbp_dir = os.path.join(orig_dir, rbp, OUTPUT_FILE)
        df_rbp = pd.read_csv(rbp_dir, sep='\t')
        df_rbp = df_rbp.dropna(axis=0)
        query = 'sequence.str.match("([ATGC]{'
        query += str(args.kmer)
        query += '}\s)+")'
        df_rbp = df_rbp.query(query)
        df_rbp = df_rbp.drop_duplicates(subset='sequence')
        pos = df_rbp[df_rbp['label']==1]
        if len(pos)>args.max_num:
            pos = pos.sample(n=args.max_num, random_state=args.random_seed)
        neg = df_rbp[df_rbp['label']==0]
        if len(neg)>args.max_num:
            neg = neg.sample(n=args.max_num, random_state=args.random_seed)
        # print(len(df_rbp), len(pos), len(neg))
        test_pos = pos.sample(frac=args.test_ratio, random_state=args.random_seed)
        test_neg = neg.sample(frac=args.test_ratio, random_state=args.random_seed)
        tr_pos = pos[~pos.sequence.isin(test_pos.sequence)].dropna()
        tr_neg = neg[~neg.sequence.isin(test_neg.sequence)].dropna()
        # print(len(test_pos), len(test_neg), len(tr_pos), len(tr_neg))
        df_test = pd.merge(test_pos, test_neg, how='outer').sample(frac=1, random_state=args.random_seed)
        # print(len(df_test))
        eval_pos = tr_pos.sample(frac=args.test_ratio, random_state=args.random_seed)
        eval_neg = tr_neg.sample(frac=args.test_ratio, random_state=args.random_seed)
        train_pos = tr_pos[~tr_pos.sequence.isin(eval_pos.sequence)].dropna()
        train_neg = tr_neg[~tr_neg.sequence.isin(eval_neg.sequence)].dropna()
        # print(len(eval_pos), len(eval_neg), len(train_pos), len(train_neg))
        df_eval = pd.merge(eval_pos, eval_neg, how='outer').sample(frac=1, random_state=args.random_seed)
        df_train = pd.merge(train_pos, train_neg, how='outer').sample(frac=1, random_state=args.random_seed)
        one_sequence_list.append([rbp, len(train_pos), len(train_neg), len(eval_pos), len(eval_neg), len(test_pos), len(test_neg), len(pos), len(neg)])
        len_sequence_list.extend(one_sequence_list)
        df_nontrain = df_rbp[~df_rbp.sequence.isin(df_train.sequence)].dropna()

        test_dir = os.path.join(orig_dir, rbp, 'test_sample_finetune')
        train_dir = os.path.join(orig_dir, rbp, 'training_sample_finetune')
        nontrain_dir = os.path.join(orig_dir, rbp, "nontraining_sample_finetune")
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)
        if not os.path.isdir(nontrain_dir):
            os.makedirs(nontrain_dir)
        test_df_path = os.path.join(test_dir, "dev.tsv")
        df_test.to_csv(test_df_path, sep='\t', index=False)
        eval_df_path = os.path.join(train_dir, "dev.tsv")
        df_eval.to_csv(eval_df_path, sep='\t', index=False)
        train_df_path = os.path.join(train_dir, "train.tsv")
        df_train.to_csv(train_df_path, sep='\t', index=False)
        nontrain_df_path = os.path.join(nontrain_dir, "dev.tsv")
        df_nontrain.to_csv(nontrain_df_path, sep='\t', index=False)

        '''
        memory_limit = 5000
        if len(df_train) > memory_limit:
            num_folders = len(df_train)//memory_limit
            if len(df_train)%memory_limit > 0:
                num_folders += 1
            print('kmer:', kmer, 'rbp:', rbp, 'num samples: ', len(df_train), 'num dirs: ', num_folders)
            for i in range(num_folders):
                train_subdir = train_dir + '/part' + str(i+1)
                start = i*memory_limit
                end = (i+1)*memory_limit
                if end > len(df_train):
                    end = len(df_train)
                if not os.path.isdir(train_subdir):
                    os.makedirs(train_subdir)
                df_train_new = df_train[start:end].copy()
                train_df_path = train_subdir + '/train.tsv'
                df_train_new.to_csv(train_df_path, sep='\t', index=False)
        '''

def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--path_to_benchmark",
        default=None,
        type=str,
        required=True,
        help="path to the benchmark file",
    )
    parser.add_argument(
        "--path_to_output",
        default=None,
        type=str,
        required=True,
        help="path to the output directory",
    )
    parser.add_argument(
        "--kmer",
        default=3,
        type=int,
        required=False,
        help="kmer of the output file",
    )
    parser.add_argument(
        "--max_num",
        default=15000,
        type=int,
        required=False,
        help="maximum number of samples to retrieve",
    )
    parser.add_argument(
        "--test_ratio",
        default=0.2,
        type=float,
        required=False,
        help="ratio of test data",
    )
    parser.add_argument(
        "--random_seed",
        default=0,
        type=int,
        required=False,
        help="seed number for random sampling",
    )
    
    args = parser.parse_args()
    
    #createkmers(args)
    preprocess(args)
    
    return
        
if __name__ == "__main__":
    main()