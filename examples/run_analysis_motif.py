import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../motif'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import argparse
import glob
import json
import logging
import os
import re
import shutil
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from transformers_DNABERT import (
    WEIGHTS_NAME,
    BertConfig,
    BertForSequenceClassification,
    BertForLongSequenceClassification,
    BertForLongSequenceClassificationCat,
    BertTokenizer,
    DNATokenizer,
)
from transformers_DNABERT import glue_compute_metrics as compute_metrics
from transformers_DNABERT import glue_convert_examples_to_features as convert_examples_to_features
from transformers_DNABERT import glue_output_modes as output_modes
from transformers_DNABERT import glue_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning) 


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "dna": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "dnalong": (BertConfig, BertForLongSequenceClassification, DNATokenizer),
    "dnalongcat": (BertConfig, BertForLongSequenceClassificationCat, DNATokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
}

TOKEN_ID_GROUP = ["bert", "dnalong", "dnalongcat"] 

MASK_LIST = {
    "3": [-1, 1],
    "4": [-1, 1, 2],
    "5": [-2, -1, 1, 2],
    "6": [-2, -1, 1, 2, 3]
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.predict_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def get_labels(args, num_start=0, num_end=-1):
    
    labels = np.load(os.path.join(args.data_dir, 'dev.tsv'))
    labels = labels['label'].to_list()[num_start:num_end]
    
    return labels

def analyze_motif(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)    

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        # evaluate = False if args.visualize_train else True
        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)
        num_samples = len(pred_dataset)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Extracting attention scores for motif analysis {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
            
        attention_scores = np.zeros([len(pred_dataset), args.max_seq_length])
        
        for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                # attention.shape : (batch_size, num_head, token_length_after, token_length_before)
                
                attention = outputs[-1][-1][:, :, 0, :]
                # attention.shape : (batch_size, num_head, token_length_before)
                attention = torch.sum(attention, dim=1)
                # attention.shape : (batch_size, token_length_before)
                
                attention_scores[index*batch_size:index*batch_size+len(batch[0]),:] = attention.cpu().numpy()
                
        logger.info("saving file: attention_scores to atten.npy (%s)", attention_scores.shape)
        np.save(os.path.join(args.predict_dir, "atten.npy"), attention_scores)
    return


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    """
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            ),
        )
    """
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            str(args.max_seq_length),
            str(task),
            ),
        )
        
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )

        print("finish loading examples")

        # params for convert_examples_to_features
        max_length = args.max_seq_length
        pad_on_left = bool(args.model_type in ["xlnet"])
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0


        if args.n_process == 1:
            features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_length,
            output_mode=output_mode,
            pad_on_left=pad_on_left,  # pad on the left for xlnet
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,)

        else:
            n_proc = int(args.n_process)
            if evaluate:
                n_proc = max(int(n_proc/4),1)
            print("number of processes for converting feature: " + str(n_proc))
            p = Pool(n_proc)
            indexes = [0]
            len_slice = int(len(examples)/n_proc)
            for i in range(1, n_proc+1):
                if i != n_proc:
                    indexes.append(len_slice*(i))
                else:
                    indexes.append(len(examples))
           
            results = []
            
            for i in range(n_proc):
                results.append(p.apply_async(convert_examples_to_features, args=(examples[indexes[i]:indexes[i+1]], tokenizer, max_length, None, label_list, output_mode, pad_on_left, pad_token, pad_token_segment_id, True,  )))
                print(str(i+1) + ' processor started !')
            
            p.close()
            p.join()

            features = []
            for result in results:
                features.extend(result.get())
                    

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--n_process",
        default=2,
        type=int,
        help="number of processes used for data process",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    
    # Other parameters
    parser.add_argument(
        "--visualize_data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--result_dir",
        default=None,
        type=str,
        help="The directory where the dna690 and mouse will save results.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--predict_dir",
        default=None,
        type=str,
        help="The output directory of predicted result. (when do_predict)",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--do_cache", action="store_true", help="whether to only cache train.tsv and dev.tsv file")
    parser.add_argument("--do_analyze_motif", action="store_true", help="Whether to analyze rnastructure")
    
    parser.add_argument("--output_visimage", action="store_true", help="generate visualized images during attention analysis")
    parser.add_argument("--rbp_name", default="", type=str, help="name of rbp to label the output image")
    
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--per_gpu_pred_batch_size", default=8, type=int, help="Batch size per GPU/CPU for prediction.",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the predict directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--visualize_models", type=int, default=None, help="The model used to do visualization. If None, use 3456.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")


    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")


    args = parser.parse_args()

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]
    if (
        os.path.exists(args.predict_dir)
        and os.listdir(args.predict_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.predict_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if not args.do_cache:
        # config.output_hidden_states = True
        config.output_attentions = True

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    logger.info('finish loading model')

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    
    # only cache examples
    if args.do_cache:
        filelist = os.listdir(args.data_dir)
        if "train.tsv" in filelist:
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
            logger.info("cached: %s", args.data_dir)
        if "dev.tsv" in filelist:
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
            logger.info("cached: %s", args.data_dir)

    elif args.do_analyze_motif:
        kmer = 3
        analyze_motif(args, model, tokenizer, prefix="", kmer=kmer)
        logger.info("finished extracting attention scores for motif analysis")


    return


if __name__ == "__main__":
    main()
