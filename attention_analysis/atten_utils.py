import os
import pandas as pd
import numpy as np
import subprocess
import re
import sys
sys.path.append('../motif')
sys.path.append('../attention_analysis')
from motif_utils import kmer2seq
import lib_forgi

FILENAME_BASE_PAIRING_PROB = 'base_pairing_prob.txt'
FILENAME_MEA = "mea.txt"

"""
entity_lookup = {'f': 'dangling start',
                     't': 'dangling end',
                     'i': 'internal loop',
                     'h': 'hairpin loop',
                     'm': 'multi loop',
                     's': 'stem'
                     }
"""

def create_fasta_from_tsv(path_to_dev, num_start, num_end, path_to_fasta=None, training=False):
    filename = os.path.join(path_to_dev, "dev.tsv")
    if training:
        filename = os.path.join(path_to_dev, "train.tsv")
    seq_data = pd.read_csv(filename, sep='\t')
    
    assert('sequence' in seq_data.columns)
    seq_data = seq_data[num_start:num_end]
    
    new_lines = ''
    for i, seq in enumerate(seq_data['sequence']):
        new_lines += '>sequence number ' + str(i) + '\n'
        origline = kmer2seq(seq)
        new_lines += origline + '\n'

    new_path = os.path.join(path_to_dev, "dev_bpp.fasta")
    if path_to_fasta:
        new_path = os.path.join(path_to_fasta, "dev_bpp.fasta")
    if training:
        new_path = os.path.join(path_to_dev, "train_bpp.fasta")
        
    with open(new_path, 'w') as f:
        f.write(new_lines)
    
    return

def execute_linearpartition(path_to_file, path_to_linearpartition, training=False):
    path_to_fasta = os.path.join(path_to_file, "dev_bpp.fasta")
    filename = 'dev_' + FILENAME_BASE_PAIRING_PROB
    if training:
        filename = 'train_' + FILENAME_BASE_PAIRING_PROB
        path_to_fasta = os.path.join(path_to_file, "train_bpp.fasta")
    path_to_bpp = os.path.join(path_to_file, filename)
    if not 'linearpartition' in path_to_linearpartition:
        path_to_linearpartition = os.path.join(path_to_linearpartition, "linearpartition")

    if os.path.isfile(path_to_bpp):
        command = 'rm {}'.format(path_to_bpp)
        subprocess.run(command, shell=True)
    command = 'cat {} | {} -o {}'.format(path_to_fasta, path_to_linearpartition, path_to_bpp)
    process_result = subprocess.run(command, shell=True)
    if process_result==1:
        print('ERROR: in execute_linearpartition for {}'.format(path_to_file))
    
    return

def get_base_pairing_prob(path_to_file, bpp_shape, training=False):
    filename = 'dev_' + FILENAME_BASE_PAIRING_PROB
    if training:
        filename = 'train_' + FILENAME_BASE_PAIRING_PROB
        
    path_to_bpp = os.path.join(path_to_file, filename)
    bpp_matrix = np.zeros(bpp_shape)
    
    with open(path_to_bpp, 'r') as f:
        count_seq = -1
        count_line = 0
        for line in f:
            count_line += 1
            if '>' in line:
                count_seq += 1
            elif '.' in line:
                values = line.split()
                i = int(values[0])-1
                j = int(values[1])-1
                value = float(values[2])
                bpp_matrix[count_seq, i, j] = value
                bpp_matrix[count_seq, j, i] = value
    
    return bpp_matrix

def tokenize_base_pairing_prob(bpp_matrix, kmer):
    #shape = [bpp_matrix.shape[0], bpp_matrix.shape[1]-kmer+1+2, bpp_matrix.shape[1]-kmer+1+2]
    shape = [bpp_matrix.shape[0], bpp_matrix.shape[1]-kmer+1, bpp_matrix.shape[1]-kmer+1]
    tokenized_bpp_matrix = np.zeros(shape)
    #batch_size, token_length, token_length(including [CLS] and [SEP])
    
    for num, matrix in enumerate(bpp_matrix):
        for i in range(shape[1]):
            for j in range(shape[2]):
                tokenized_bpp_matrix[num, i, j] = np.sum(matrix[i:i+kmer, j:j+kmer])
    
    
    return tokenized_bpp_matrix

def get_mea_structures(path_to_file, path_to_linearpartition, training=False):
    #filename = 'dev_' + FILENAME_MEA
    path_to_fasta = os.path.join(path_to_file, "dev_bpp.fasta")
    if training:
        #filename = 'train_' + FILENAME_MEA
        path_to_fasta = os.path.join(path_to_file, "train_bpp.fasta")
        
    #path_to_mea = os.path.join(path_to_file, filename)
    
    if not 'linearpartition' in path_to_linearpartition:
        path_to_linearpartition = os.path.join(path_to_linearpartition, "linearpartition")
    
    #if os.path.isfile(path_to_mea):
    #    command = 'rm {}'.format(path_to_mea)
    #    subprocess.run(command, shell=True)
    # command = 'cat {} | {} -m > {}'.format(path_to_fasta, path_to_linearpartition, path_to_mea)
    command = 'cat {} | {} -m'.format(path_to_fasta, path_to_linearpartition)
    # print(command)
    #process_result = 1
    #with open(path_to_mea, 'w') as output_file:
    #    process_result = subprocess.run(command, shell=True, stdout=output_file)
    # print(command)
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, check=True)
    # process_result = subprocess.run(command, shell=True)
    # print("process done")
    
    structure_seqs = np.array([])
    if output.returncode == 1:
        print('ERROR: in execute_linearpartition_mea for {}'.format(path_to_file))
    else:
        def make_node_set(numbers):
            numbers = list(map(int, numbers))
            ans = set()
            while len(numbers) > 1:
                a, b = numbers[:2]
                numbers = numbers[2:]
                for n in range(a - 1, b):
                    ans.add(n)  # should be range a,b+1 but the biologists are weired
            return ans
        
        #structures = []
        #with open(path_to_mea, 'r') as f:
        #    structures = f.readlines()
        structures = output.stdout.decode().strip().split('\n')
        
        count = 0
        entity_lookup = {'f': 0, 't': 1, 'i': 2, 'h': 3, 'm': 4, 's': 5}
        # f: 'dangling start', 't': 'dangling end', 'i': 'internal loop', 'h': 'hairpin loop', 'm': 'multi loop', 's': 'stem'
        for structure in structures:
            #structure = structure.strip()
            if re.fullmatch("[\.\(\)]+", structure):
                #print(structure)
                bg = lib_forgi.BulgeGraph()
                bg.from_dotbracket(structure, None)
                forgi = bg.to_bg_string()
                structure_sequence = np.zeros([1, 6, len(structure)])
                for line in forgi.split('\n')[:-1]:
                    # if the line starts with 'define' we know that annotation follows...
                    if line[0] == 'd':
                        l = line.split()
                        # first we see the type
                        entity = l[1][0]
                        # then we see a list of nodes of that type.
                        entity_index = entity_lookup[entity]
                        for n in make_node_set(l[2:]):
                            structure_sequence[0,entity_index,n] = 1
                            
                if len(structure_seqs)==0:
                    structure_seqs = structure_sequence
                else:
                    structure_seqs = np.concatenate([structure_seqs, structure_sequence])
    return structure_seqs