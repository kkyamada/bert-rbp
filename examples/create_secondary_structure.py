import os
import sys
import pandas as pd
import numpy as np
import subprocess
import re
sys.path.append('../motif')
sys.path.append('../attention_analysis')
from motif_utils import kmer2seq
from atten_utils import create_fasta_from_tsv
from atten_utils import get_mea_structures
import argparse

entity_lookup = {'f': 0, 't': 1, 'i': 2, 'h': 3, 'm': 4, 's': 5}
entity_lookup = {v:k for k,v in entity_lookup.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data directory",
    )
    parser.add_argument(
        "--path_to_linearpartition",
        default=None,
        type=str,
        required=True,
        help="directory to the linearpartition",
    )
    parser.add_argument(
        "--rbp",
        default=None,
        type=str,
        required=True,
        help="RBP data to process",
    )
    parser.add_argument(
        "--data_dir_suffix",
        default="",
        type=str,
        required=False,
        help="The input data directory after RBP name",
    )
    parser.add_argument(
        "--num_each",
        default=1000,
        type=int,
        required=False,
        help="number of samples to process for each for-loop",
    )
    args = parser.parse_args()
    ORIG_PATH = args.data_dir
    PATH2 = args.data_dir_suffix
    PATH_TO_LINEARPARTITION = args.path_to_linearpartition
    NUM_EACH = args.num_each
    rbp = args.rbp
    
    print(args)
    
    path = os.path.join(ORIG_PATH, rbp, PATH2)
    df = pd.read_csv(os.path.join(path, "dev.tsv"), sep="\t")
    
    tmp_num = len(df)//NUM_EACH + 1
    print("number of sequences:", len(df))
    
    all_list = []
    last_progress = 0
    for j in range(tmp_num):
        num_start = j * NUM_EACH
        num_end = (j+1) * NUM_EACH
        if num_end >= len(df):
            num_end = -1
            
        create_fasta_from_tsv(path,
                              num_start=num_start,
                              num_end=num_end,
                              path_to_fasta=None,
                              training=False)
        structures = get_mea_structures(path, PATH_TO_LINEARPARTITION, training=False)
        structure_list = np.full((structures.shape[0], structures.shape[-1]), "")
        for i in range(6):
            structure_list[np.where(structures[:,i,:]==1)] = entity_lookup[i]
        structure_list = structure_list.tolist()
        for i in range(len(structure_list)):
            structure_list[i] = "".join(structure_list[i])
        
        all_list.extend(structure_list)
        progress = len(all_list)/len(df) // 0.1
        if last_progress < progress:
            print("finished processing {}/{} sequences".format(len(all_list), len(df)))
        last_progress = progress
        
        
    save_path = os.path.join(ORIG_PATH, rbp, PATH2, "dev_structure.txt")
    print("saving to ...", save_path)
    all_list = "\n".join(all_list) + "\n"
    with open(save_path, "w") as f:
        f.write(all_list)
    return

if __name__ == "__main__":
    main()
