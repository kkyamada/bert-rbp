# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modeified by Keisuke YAMADA 11/17/2020-
# run_finetune-Copy1 -> Copy2 enabled attention analysis
# run_finetune-Copy2 -> Copy3 enabled do_cache
# run_finetune-Copy3 -> Copy4 enabled do_train_from_scratch
# run_finetune-Copy4 -> Copy5 editted attention analysis
# run_finetune-Copy5 -> Copy6 added structural probe (inner product->logistic regression for predicting base pairing prob matrix)
# run_finetune-Copy6 -> Copy7 added attention probe
# run_finetune-Copy7 -> Copy8 added hyperparameter tuning(evaluation) for structural probing
# run_finetune-Copy8 -> Copy9 added hyperparameter tuning(evaluation) for attention probing
# run_finetune-Copy9 -> Copy10 added GC content analysis
# run_finetune-Copy10 -> Copy11 added region type analysis updated GC analysis
# run_finetune-Copy11 -> Copy12 updated motif analysis
# run_finetune-Copy12 -> Copy13 added attention rollout calculation
# run_finetune-Copy13 -> Copy14 updated analyze_regiontype
# run_finetune-Copy14 -> Copy15 addded analyze_structures
# run_finetune-Copy15 -> Copy16 added analyze_regionboundary
# run_finetune-Copy16 -> Copy17 added positional_effects and modified analyze_rnastructure
# run_finetune-Copy17 -> Copy18 added positional_effects_dir
# run_finetune-Copy18 -> Copy19 modified analyze_regionboundary
# run_finetune-Copy19 -> Copy20 added analyze_rnastructure_specific
# run_finetune-Copy20 -> Copy22 added analyze_regionboundary_specific
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import os
import sys
#sys.path.append('../')
#sys.path.append('../examples')
sys.path.append(os.path.join(os.path.dirname(__file__), '../attention_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../motif'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from atten_utils import create_fasta_from_tsv
from atten_utils import execute_linearpartition
from atten_utils import get_base_pairing_prob
from atten_utils import tokenize_base_pairing_prob
from atten_utils import get_mea_structures
import pandas as pd

import argparse
import glob
import json
import logging
import re
import shutil
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers_DNABERT import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertForLongSequenceClassification,
    BertForLongSequenceClassificationCat,
    BertTokenizer,
    DNATokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers_DNABERT import glue_compute_metrics as compute_metrics
# from transformers import glue_compute_metrics as compute_metrics
from transformers_DNABERT import glue_convert_examples_to_features as convert_examples_to_features
from transformers_DNABERT import glue_output_modes as output_modes
from transformers_DNABERT import glue_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "dna": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "dnalong": (BertConfig, BertForLongSequenceClassification, DNATokenizer),
    "dnalongcat": (BertConfig, BertForLongSequenceClassificationCat, DNATokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}
                    
TOKEN_ID_GROUP = ["bert", "dnalong", "dnalongcat", "xlnet", "albert"] 

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

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

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.local_rank==-1 and args.n_gpu>1:
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size * args.n_gpu)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent*t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1,args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    #if args.fp16:
    #    try:
    #        from apex import amp
    #    except ImportError:
    #        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    best_auc = 0
    last_auc = 0
    stop_count = 0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in TOKEN_ID_GROUP else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                #logger.info('loss:{}'.format(loss))
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)


                        if args.task_name == "dna690":
                            # record the best auc
                            if results["auc"] > best_auc:
                                best_auc = results["auc"]

                        if args.early_stop != 0:
                            # record current auc to perform early stop
                            if results["auc"] < last_auc:
                                stop_count += 1
                            else:
                                stop_count = 0

                            last_auc = results["auc"]
                            
                            if stop_count == args.early_stop:
                                logger.info("Early stop")
                                return global_step, tr_loss / global_step


                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.task_name == "dna690" and results["auc"] < best_auc:
                        continue
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    if args.task_name != "dna690":
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix="", evaluate=True):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)
    if args.task_name[:3] == "dna":
        softmax = torch.nn.Softmax(dim=1)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
            

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        probs = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                #if nb_eval_steps < 10:
                #    logger.info('inputs:{}'.format(inputs['input_ids']))
                
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        #logger.info('outputs: {}'.format(outputs))
        #logger.info('logits: {}'.format(logits))
        #logger.info('preds1: {}'.format(preds))
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            if args.task_name[:3] == "dna" and args.task_name != "dnasplice":
                if args.do_ensemble_pred:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                else:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
            elif args.task_name == "dnasplice":
                probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        if args.do_ensemble_pred:
            result = compute_metrics(eval_task, preds, out_label_ids, probs[:,1])
        else:
            result = compute_metrics(eval_task, preds, out_label_ids, probs)
        results.update(result)
        
        if args.task_name == "dna690":
            eval_output_dir = args.result_dir
            if not os.path.exists(args.result_dir): 
                os.makedirs(args.result_dir)
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        if args.do_predict:
            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results_prediction.txt")
        with open(output_eval_file, "a") as writer:

            if args.task_name[:3] == "dna":
                eval_result = args.data_dir.split('/')[-1] + " "
            else:
                eval_result = ""

            #logger.info('preds: {}'.format(preds))
            #logger.info('labels: {}'.format(out_label_ids))
            #logger.info('probs: {}'.format(probs))
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                eval_result = eval_result + str(result[key])[:5] + " "
            writer.write(eval_result + "\n")

    if args.do_ensemble_pred:
        return results, eval_task, preds, out_label_ids, probs
    else:
        return results

def predict(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    predictions = {}
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=True)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(pred_dataloader, desc="Predicting"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                _, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            if args.task_name[:3] == "dna" and args.task_name != "dnasplice":
                if args.do_ensemble_pred:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                else:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
            elif args.task_name == "dnasplice":
                probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        if args.do_ensemble_pred:
            result = compute_metrics(pred_task, preds, out_label_ids, probs[:,1])
        else:
            result = compute_metrics(pred_task, preds, out_label_ids, probs)
            
        pred_output_dir = args.predict_dir
        if not os.path.exists(pred_output_dir):
               os.makedir(pred_output_dir)
        output_pred_file = os.path.join(pred_output_dir, "pred_results.npy")
        logger.info("***** Pred results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        np.save(output_pred_file, probs)

def extract(args, model, tokenizer, prefix="", evaluate=True):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    predictions = {}
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        preds = None
        out_label_ids = None
        embedded_output = []
        for batch in tqdm(pred_dataloader, desc="Predicting"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                #if embedded_output == None:
                #    embedded_output = list(outputs[2])
                #else:
                #    for i in range(len(embedded_output)):
                #        embedded_output[i] = torch.cat([embedded_output[i], outputs[2][i]], axis=0)
                if len(embedded_output) == 0:
                    #embedded_output.append(outputs[2][0])
                    embedded_output.append(outputs[2][args.extract_layer])
                else:
                    #embedded_output[0] = torch.cat([embedded_output[0], outputs[2][0]], axis=0)
                    embedded_output[0] = torch.cat([embedded_output[0], outputs[2][args.extract_layer]], axis=0)
        
        #torch.save(embedded_output[0], args.predict_dir + '/emb.pt')
        if evaluate:
            torch.save(embedded_output[0], args.predict_dir + '/dev.pt')
        else:
            torch.save(embedded_output[0], args.predict_dir + '/train.pt')
        return

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed).unsqueeze(0)

def analyze_positional_effects(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''

        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)
        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12
        
        # count = 0
        # entity_lookup = {'f': 0, 't': 1, 'i': 2, 'h': 3, 'm': 4, 's': 5}
        positional_effects = np.zeros([num_layer, num_head, args.max_seq_length])
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
                # logger.info('outputs: {}'.format(len(outputs))) #
                # logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                attention = torch.stack(outputs[-1], 1)[:,:,:,:,:]
                attention = torch.sum(attention[:, :, :, :, :], dim=-2)

                # attention.shape = [batch_size, num_layer, num_head, token_length_after(query), token_length_before(key, value)]
                # attention = torch.sum(attention, dim=-2)
                # attention.shape = [batch_size, num_layer, num_head, token_length_before(key, value)]
                # if num_layer==1:
                #    attention = attention[:, args.vis_layer-1:args.vis_layer, :, :, :]
                #    if num_head==1:
                #        attention = attention[:,:,args.vis_head-1:args.vis_head,:,:]

                # attention.shape = [batch_size, num_layer, num_head, token_length_after]
                # attention[:,:,:, :(kmer-1),:] = 0.0
                # attention[:,:,:, :,:(kmer-1)] = 0.0
                # attention[:,:,:,-(kmer-1):,:] = 0.0
                # attention[:,:,:,:,-(kmer-1):] = 0.0
                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                
                positional_effects = positional_effects + torch.sum(attention, dim=0).cpu().detach().numpy()

    return positional_effects

def get_structures(args, kmer, num_start=0, num_end=-1, binary=True, tokenize=True, training=False):
    # -2 includes [CLS] and [SEP] tokens
    path_tmp = args.data_dir

    if args.path_to_fasta:
        create_fasta_from_tsv(args.data_dir, num_start, num_end, args.path_to_fasta, training=training)
        path_tmp = args.path_to_fasta
    else:
        create_fasta_from_tsv(path_tmp, num_start, num_end, training=training)
    
    structure_seqs = get_mea_structures(path_tmp, args.path_to_linearpartition, training=training)
    """
    entity_lookup = {'f': 'dangling start',
                     't': 'dangling end',
                     'i': 'internal loop',
                     'h': 'hairpin loop',
                     'm': 'multi loop',
                     's': 'stem'}
    """
    structure_sequences = structure_seqs
    if tokenize:
        structure_sequences = np.zeros([structure_seqs.shape[0], structure_seqs.shape[1], structure_seqs.shape[-1]-kmer+3])
        for i in range(kmer):
            structure_sequences[:, :, 1:-1] = structure_sequences[:, :, 1:-1] + structure_seqs[:, :, i:i+structure_seqs.shape[-1]-kmer+1]
    
    if binary:
        structure_sequences = np.where(structure_sequences > 0, 1, 0)
    
    return structure_sequences

def analyze_rnastructure_specific(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)
    
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        
        structures_and_heads = []
        if args.specific_heads:
            lines = re.findall("[0-9]+[,-][0-9]+", args.specific_heads)
            for structure_and_head in lines:
                structure_and_head = structure_and_head.split(",")
                structure_and_head = list(map(int, structure_and_head))
                structure_and_head = list(np.array(structure_and_head) - 1)
                structures_and_heads.append(structure_and_head)
        else:
            textfile = os.path.join(pred_output_dir, "analyze_rnastructure.txt")
            with open(textfile, 'r') as f:
                for line in f:
                    structure_and_head = []
                    if re.search("max_head: [0-9\-]+", line):
                        max_head = re.findall("max_head: [0-9\-]+", line)[0]
                        structure_and_head = [int(i) for i in re.findall("[0-9]+", max_head)]
                        structures_and_heads.append(structure_and_head)
            
        tsv_file_original = os.path.join(pred_output_dir, "analyze_rnastructure_specific_layer{}_head{}.tsv")
        for structure_and_head in structures_and_heads:
            tsv_file = tsv_file_original.format(structure_and_head[0]+1, structure_and_head[1]+1)
            with open(tsv_file, "w") as f:
                f.write("structure\tattention\n")
        logger.info("structures_and_heads: {}".format(structures_and_heads))
        
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        
        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12
        
        positional_effects = torch.ones(num_layer, num_head, args.max_seq_length)
        if args.correct_position_bias:
            positional_effects_dir = os.path.join(args.positional_effects_dir, "positional_effects.npy")
            if os.path.isfile(positional_effects_dir):
                logger.info("positional_effects_dir: {}".format(positional_effects_dir))
                positional_effects = np.load(positional_effects_dir) / 100000
                positional_effects = np.where(positional_effects==0, np.min(positional_effects), positional_effects)
                positional_effects = torch.tensor(positional_effects)
        
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
                # logger.info('outputs: {}'.format(len(outputs))) #
                # logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                attention = torch.stack(outputs[-1], 1)[:,:,:,:,:]
                #if num_layer==1:
                #    attention = attention[:, args.vis_layer-1:args.vis_layer, :, :, :]
                #    if num_head==1:
                #        attention = attention[:,:,args.vis_head-1:args.vis_head,:,:]
                attention = torch.sum(attention, dim=-2)
                #logger.info("attention.shape {}".format(attention.shape))
                # attention.shape = [batch_size, 12, 12, token_length_before(key, value)]

                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                structure_sequences = get_structures(args, kmer, num_start, num_end, binary=True, tokenize=True)
                #structure_sequences_inv = structure_sequences * -1 + 1
                #structure_sequences_inv[:,:,0] = 0
                #structure_sequences_inv[:,:,-1] = 0
                
                # logger.info('attention_map: {}'.format(attention.shape))
                # logger.info('annotations: {}'.format(annotations.shape))
                # annotations.shape : batch, token_length
                '''
                for i, attention_sample in enumerate(attention):
                    annotation_all_matrix = annotation_all_matrix + torch.sum(attention_sample, dim=-1).cpu().detach().numpy()
                    for layer in range(num_layer):
                        for head in range(num_head):
                            annotation_matrix[layer, head] = annotation_matrix[layer, head] +\
                                                             np.sum(annotations[i] * attention_sample[layer, head].cpu().detach().numpy())
                            #if i==1 and layer==2 and head==0:
                            #    logger.info('annotation_all_matrix: {}, annotation_matrix: {}'.format(\
                            #                 attention_sample.cpu().detach().numpy()[layer,head],\
                            #                 annotations[i]))
                '''
                
                
                    # structure_tmp = torch.tensor(structure_sequences[:,entity,:]).repeat(num_layer,num_head,1,1).transpose(0,1).transpose(0,2)
                    # structure_tmp.shape : (batch_size, num_layer, num_head, token_length_after)
                    # structure_tmp = structure_tmp / positional_effects.repeat(structure_tmp.shape[0], 1, 1, 1)
                    # attention_tmp = attention / positional_effects.repeat(structure_tmp.shape[0], 1, 1, 1)
                    # attention_to_entity.shape : batch_size, num_layer, num_head, token_length_after
                for structure_and_head in structures_and_heads:
                    for entity in range(structure_sequences.shape[1]):
                        tsv_file = tsv_file_original.format(structure_and_head[0]+1, structure_and_head[1]+1)
                        structure_tmp = structure_sequences[:,entity,:]
                        attention_to_entity = attention[:, structure_and_head[0], structure_and_head[1], :]
                        if args.correct_position_bias:
                            attention_to_entity = attention_to_entity / positional_effects[structure_and_head[0], structure_and_head[1]].repeat(structure_tmp.shape[0], 1)
                        total_attentions = torch.sum(attention_to_entity[:, 1:-1], dim=-1).repeat(attention_to_entity.shape[-1],1).transpose(0,1)
                        attention_to_entity = (attention_to_entity * structure_tmp) / total_attentions
                        #logger.info('attention_to_entity.shape: {}'.format(attention_to_entity.shape))
                        attention_to_entity = attention_to_entity[np.where(structure_tmp > 0)].cpu().detach().numpy()[:,np.newaxis]
                        
                        attention_to_entity = pd.DataFrame(np.concatenate([np.full([len(attention_to_entity), 1], entity), attention_to_entity], axis=1),\
                                                           columns=["structure", "attention"])
                        attention_to_entity.to_csv(tsv_file, sep="\t", index=False, header=False, mode="a")
    return

def analyze_rnastructure(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''

        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)
        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12
    
        #'''
        positional_effects = torch.ones(num_layer, num_head, args.max_seq_length)
        if args.correct_position_bias:
            positional_effects_dir = os.path.join(args.positional_effects_dir, "positional_effects.npy")
            if os.path.isfile(positional_effects_dir):
                logger.info("positional_effects_dir: {}".format(positional_effects_dir))
                positional_effects = np.load(positional_effects_dir) / 100000
                positional_effects = np.where(positional_effects==0, np.min(positional_effects), positional_effects)
                positional_effects = torch.tensor(positional_effects)
        #'''
        
        # count = 0
        # entity_lookup = {'f': 0, 't': 1, 'i': 2, 'h': 3, 'm': 4, 's': 5}
        rnastructure_matrix = np.zeros([6, num_layer, num_head])
        rnastructure_count = np.zeros(6)
        rnastructure_matrix_negative = np.zeros([6, num_layer, num_head])
        rnastructure_count_negative = np.zeros(6)
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
                # logger.info('outputs: {}'.format(len(outputs))) #
                # logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                attention = torch.stack(outputs[-1], 1)[:,:,:,:,:]
                attention = torch.sum(attention[:, :, :, :, :], dim=-2)

                # attention.shape = [batch_size, num_layer, num_head, token_length_after(query), token_length_before(key, value)]
                # attention = torch.sum(attention, dim=-2)
                # attention.shape = [batch_size, num_layer, num_head, token_length_before(key, value)]
                # if num_layer==1:
                #    attention = attention[:, args.vis_layer-1:args.vis_layer, :, :, :]
                #    if num_head==1:
                #        attention = attention[:,:,args.vis_head-1:args.vis_head,:,:]

                # attention.shape = [batch_size, num_layer, num_head, token_length_after]
                # attention[:,:,:, :(kmer-1),:] = 0.0
                # attention[:,:,:, :,:(kmer-1)] = 0.0
                # attention[:,:,:,-(kmer-1):,:] = 0.0
                # attention[:,:,:,:,-(kmer-1):] = 0.0
                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                structure_sequences = get_structures(args, kmer, num_start, num_end, binary=True, tokenize=True)
                structure_sequences_inv = structure_sequences * -1 + 1
                structure_sequences_inv[:,:,0] = 0
                structure_sequences_inv[:,:,-1] = 0

                # structure_sequences.shape : (batch_size, 6, token_length)
                # logger.info("structure_sequences.shape: {}".format(structure_sequences.shape))
                """    
                entity_lookup = {'f': 0, 't': 1, 'i': 2, 'h': 3, 'm': 4, 's': 5}
                if len(args.target_entity)==1:
                    index = entity_lookup[args.target_entity]
                    structure_sequences = structure_sequences[:, index, :]
                elif len(arg.target_entity) < len(entity_lookup):
                    tmp = np.zeros(structure_sequences.shape[0], structure_sequences.shape[-1])
                    for target_entity in list(args.target_entity):
                        index = entity_lookup[target_entity]
                        tmp = tmp + structure_sequences[:, index, :]
                    structure_sequences = np.where(tmp > 0, 1, tmp)
                else
                    assert False
                """

                # annotations = np.zeros([annot.shape[0], annot.shape[1], annot.shape[-1]-kmer+3])
                # for i in range(kmer):
                #    annotations[:, :, 1:-1] = annotations[:, :, 1:-1] + annot[:, :, i:i+annot.shape[-1]-kmer+1]
                
                # logger.info('attention_map: {}'.format(attention.shape))
                # logger.info('annotations: {}'.format(annotations.shape))
                '''
                for i, attention_sample in enumerate(attention):
                    annotation_all_matrix = annotation_all_matrix + torch.sum(attention_sample, dim=-1).cpu().detach().numpy()
                    for layer in range(num_layer):
                        for head in range(num_head):
                            annotation_matrix[layer, head] = annotation_matrix[layer, head] +\
                                                             np.sum(annotations[i] * attention_sample[layer, head].cpu().detach().numpy())
                            #if i==1 and layer==2 and head==0:
                            #    logger.info('annotation_all_matrix: {}, annotation_matrix: {}'.format(\
                            #                 attention_sample.cpu().detach().numpy()[layer,head],\
                            #                 annotations[i]))
                '''
                rnastructure_count = rnastructure_count + np.sum(np.sum(structure_sequences, axis=-1), axis=0)
                rnastructure_count_negative = rnastructure_count_negative + np.sum(np.sum(structure_sequences_inv, axis=-1), axis=0)

                for entity in range(structure_sequences.shape[1]):
                    structure_tmp = torch.tensor(structure_sequences[:,entity,:]).repeat(num_layer,num_head,1,1).transpose(0,1).transpose(0,2)
                    # structure_tmp.shape : (batch_size, num_layer, num_head, token_length_after)
                    #structure_tmp = structure_tmp
                    structure_tmp_inv = torch.tensor(structure_sequences_inv[:,entity,:]).repeat(num_layer,num_head,1,1).transpose(0,1).transpose(0,2)
                    #structure_tmp_inv = structure_tmp_inv
                    
                    attention_to_entity = torch.sum(torch.sum(attention * structure_tmp, dim=0), dim=-1)
                    attention_to_others = torch.sum(torch.sum(attention * structure_tmp_inv, dim=0), dim=-1)
                    
                    #attention_to_entity = torch.sum(torch.sum((attention - positional_effects.repeat(structure_tmp.shape[0], 1, 1, 1)) * structure_tmp, dim=0), dim=-1)
                    #attention_to_others = torch.sum(torch.sum((attention - positional_effects.repeat(structure_tmp_inv.shape[0], 1, 1, 1)) * structure_tmp_inv, dim=0), dim=-1)
                    
                    # attention_to_entity.shape : batch_size, num_layer, num_head, token_length_after
                    # attention_from_boundary = torch.sum(torch.sum(attentions[:, layer, head, :, :], dim=-2) * annotation_boundary, dim=0)
                    rnastructure_matrix[entity, :, :] = rnastructure_matrix[entity, :, :] + attention_to_entity.cpu().detach().numpy()
                    rnastructure_matrix_negative[entity, :, :] = rnastructure_matrix_negative[entity, :, :] + attention_to_others.cpu().detach().numpy()
        #for j in range(annotations.shape[1]):
        #    annotation_matrix[j] = annotation_matrix[j] / annotation_count[j]
        #    annotation_matrix_negative[j] = annotation_matrix_negative[j] / annotation_count_negative[j]

    return rnastructure_matrix, rnastructure_matrix_negative, rnastructure_count, rnastructure_count_negative

def get_annotations(args, kmer, num_start=0, num_end=-1, binary=True, tokenize=True, add_nonannotated=False):
    # region_type 0: 5'UTR, 1: 3'UTR, 2: exon, 3: intron, 4: CDS
    
    annot = np.load(os.path.join(args.data_dir, 'annotations.npy'))
    annot = annot[num_start:num_end, :, :]
    if args.region_type > 0:
        annot = annot[:, args.region_type-1:args.region_type, :]
    if add_nonannotated:
        nonannotated = np.sum(annot, axis=1)
        nonannotated = np.where(nonannotated > 0, -1, nonannotated) + 1
        annot = np.concatenate([annot, nonannotated[:, np.newaxis, :]], axis=1)
        
    annotations = annot
    # annotations.shape : (num of sequences, num of region types, token lengths)
    if tokenize:
        annotations = np.zeros([annot.shape[0], annot.shape[1], annot.shape[-1]-kmer+3])
        for i in range(kmer):
            annotations[:, :, 1:-1] = annotations[:, :, 1:-1] + annot[:, :, i:i+annot.shape[-1]-kmer+1]
            
    if binary:
        annotations = np.where(annotations > 0, 1, 0)

    return annotations

def get_labels(args, num_start=0, num_end=-1):
    
    labels = np.load(os.path.join(args.data_dir, 'dev.tsv'))
    labels = labels['label'].to_list()[num_start:num_end]
    
    return labels

def analyze_regionboundary_specific(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)
    
    regionboundaries = [[1,4],[0,4],[3,4],[0,3],[1,3]]
    if not args.region_boundaries == "":
        lines = re.findall("[0-9]+[,-][0-9]+", args.region_boundaries)
        for regions in lines:
            regions = regions.split(",")
            regions = list(map(int, regions))
            regions = list(np.array(structure_and_head))
            regionboundaries.append(regions)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        regiontypes = ("5'UTR", "3'UTR", "exon", "intron", "CDS", "outside")
        regions_and_heads = []
        if args.specific_heads:
            lines = re.findall("[1-9][,-][0-9]+[,-][0-9]+", args.specific_heads)
            for region_and_head in lines:
                region_and_head = region_and_head.split(",")
                region_and_head = list(map(int, region_and_head))
                regions_and_heads.append(region_and_head)
        else:
            textfile = os.path.join(pred_output_dir, "analyze_regionboundary.txt")
            regions_and_heads = []
            regiontypes = ("5'UTR", "3'UTR", "exon", "intron", "CDS")
            index_regiontypes = {"5'UTR":1, "3'UTR":2, "exon":3, "intron":4, "CDS":5}
            with open(textfile, 'r') as f:
                for line in f:
                    region_and_head = []
                    if re.search("max_head: [0-9\-]+", line):
                        max_head = re.findall("max_head: [0-9\-]+", line)[0]
                        region_and_head = [int(i) for i in re.findall("[0-9]+", max_head)]
                        regions_and_heads.append(region_and_head)
            
        tsv_file_original = os.path.join(pred_output_dir, "analyze_regionboundary_specific_all_layer{}_head{}.tsv")
        for region_and_head in regions_and_heads:
            tsv_file = tsv_file_original.format(region_and_head[0], region_and_head[1])
            with open(tsv_file, "w") as f:
                f.write("regiontype\tattention\n")
        logger.info("regions_and_heads: {}".format(regions_and_heads))
        
        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12
        
        positional_effects = torch.ones(num_layer, num_head, args.max_seq_length)
        if args.correct_position_bias:
            positional_effects_dir = os.path.join(args.positional_effects_dir, "positional_effects.npy")
            if os.path.isfile(positional_effects_dir):
                logger.info("positional_effects_dir: {}".format(positional_effects_dir))
                positional_effects = np.load(positional_effects_dir) / 100000
                positional_effects = np.where(positional_effects==0, np.min(positional_effects), positional_effects)
                positional_effects = torch.tensor(positional_effects)
        
        # count = 0
        #regionboundary_matrix = np.zeros([len(regionboundaries), num_layer, num_head])
        #regionboundary_count = np.zeros(len(regionboundaries))
        #regionboundary_matrix_negative = np.zeros([len(regionboundaries), num_layer, num_head])
        #regionboundary_count_negative = np.zeros(len(regionboundaries))
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
                # logger.info('outputs: {}'.format(len(outputs))) #
                # logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                attention = torch.stack(outputs[-1], 1)[:,:,:,:,:]
                # attention.shape = [batch_size, num_layer, num_head, token_length_after(query), token_length_before(key, value)]
                attention = torch.sum(attention, dim=-2)
                # attention.shape = [batch_size, num_layer, num_head, token_length_before(key, value)]
                #attention = torch.sum(attention, dim=-2)
                # attention.shape = [batch_size, num_layer, num_head, token_length_before(key, value)]
                #if num_layer==1:
                #    attention = attention[:, args.vis_layer-1:args.vis_layer, :, :, :]
                #    if num_head==1:
                #        attention = attention[:,:,args.vis_head-1:args.vis_head,:,:]

                # attention.shape = [batch_size, num_layer, num_head, token_length_after]
                # attention[:,:,:, :(kmer-1),:] = 0.0
                # attention[:,:,:, :,:(kmer-1)] = 0.0
                # attention[:,:,:,-(kmer-1):,:] = 0.0
                # attention[:,:,:,:,-(kmer-1):] = 0.0

                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                annotations = get_annotations(args, kmer, num_start, num_end, binary=False, tokenize=True, add_nonannotated=True)
                
                
                annotation_boundaries = np.zeros([annotations.shape[0], len(regionboundaries), annotations.shape[-1]])
                for boundary_num, regionboundary in enumerate(regionboundaries):
                    annotation_boundary = annotations[:, regionboundary[0], :]
                    annotation_boundary2 = annotations[:, regionboundary[1], :]
                    annotation_boundary = np.where(annotation_boundary==kmer, 0, annotation_boundary)
                    annotation_boundary2 = np.where(annotation_boundary2==kmer, 0, annotation_boundary2)
                    annotation_boundary = annotation_boundary * annotation_boundary2
                    annotation_boundary = np.where(annotation_boundary>0, 1, 0)
                    annotation_boundaries[:, boundary_num, :] = annotation_boundary
                #annotation_boundaries_inv = annotation_boundaries * -1 + 1
                #annotation_boundaries_inv[:,:,0] = 0
                #annotation_boundaries_inv[:,:,-1] = 0
                
                # annotation_boundaries.shape : batch, boundary_type, token_length
                
                
                # annotations = np.zeros([annot.shape[0], annot.shape[1], annot.shape[-1]-kmer+3])
                # for i in range(kmer):
                #    annotations[:, :, 1:-1] = annotations[:, :, 1:-1] + annot[:, :, i:i+annot.shape[-1]-kmer+1]
                
                # logger.info('attention_map: {}'.format(attention.shape))
                # logger.info('annotations: {}'.format(annotations.shape))
                '''
                for i, attention_sample in enumerate(attention):
                    annotation_all_matrix = annotation_all_matrix + torch.sum(attention_sample, dim=-1).cpu().detach().numpy()
                    for layer in range(num_layer):
                        for head in range(num_head):
                            annotation_matrix[layer, head] = annotation_matrix[layer, head] +\
                                                             np.sum(annotations[i] * attention_sample[layer, head].cpu().detach().numpy())
                            #if i==1 and layer==2 and head==0:
                            #    logger.info('annotation_all_matrix: {}, annotation_matrix: {}'.format(\
                            #                 attention_sample.cpu().detach().numpy()[layer,head],\
                            #                 annotations[i]))
                '''
                #regionboundary_count[0] += np.sum(annotation_boundary)
                #regionboundary_count_negative[0] += np.sum(annotation_boundary_inv)
                annotation_boundary_inv = np.sum(annotation_boundaries[:,:,1:-1], axis=1)
                annotation_boundary_inv = np.where(annotation_boundary_inv>0, 0, 1)
                
                
                for boundary_num in range(len(regionboundaries)):
                    annotation_boundary = annotation_boundaries[:,boundary_num,1:-1]
                    #annotation_boundary = torch.tensor(annotation_boundary).repeat(num_layer,num_head,1,1).transpose(0,1).transpose(0,2)
                    #annotation_boundary_inv = torch.tensor(annotation_boundary_inv).repeat(num_layer,num_head,1,1).transpose(0,1).transpose(0,2)
                    #annotation_boundary = attention * annotation_boundary
                    #annotation_boundary_inv = attention * annotation_boundary_inv
                    #batch, layer, head, token_len
                    for region_and_head in regions_and_heads:
                #for entity in range(structure_sequences.shape[1]):
                        tsv_file = tsv_file_original.format(region_and_head[0], region_and_head[1])
                        index_layer, index_head = region_and_head[0]-1, region_and_head[1]-1
                        attention_to_boundary = attention[:, index_layer, index_head, 1:-1]
                        attention_to_boundary = attention_to_boundary / torch.sum(attention_to_boundary, dim=-1).repeat(attention_to_boundary.shape[-1],1).transpose(0,1)
                        #attention_to_boundary_inv = attention_to_boundary[np.where(annotation_boundary == 0)].cpu().detach().numpy()[:, np.newaxis]
                        if boundary_num == 0:
                            attention_to_boundary_inv = attention_to_boundary[np.where(annotation_boundary_inv == 1)].cpu().detach().numpy()[:, np.newaxis]
                            attention_to_boundary_inv = pd.DataFrame(np.concatenate([np.full([len(attention_to_boundary_inv), 1], "outside"),\
                                                                                     attention_to_boundary_inv], axis=1), columns=["regiontype", "attention"])
                            attention_to_boundary_inv.to_csv(tsv_file, sep="\t", index=False, header=False, mode="a")
                        attention_to_boundary = attention_to_boundary[np.where(annotation_boundary > 0)].cpu().detach().numpy()[:, np.newaxis]
                        regiontype_title = "{}_and_{}".format(regiontypes[regionboundaries[boundary_num][0]], regiontypes[regionboundaries[boundary_num][1]])
                        attention_to_boundary = pd.DataFrame(np.concatenate([np.full([len(attention_to_boundary), 1], regiontype_title), attention_to_boundary], \
                                                                                                            axis=1), columns=["regiontype", "attention"])
                        #attention_to_boundary_inv = pd.DataFrame(np.concatenate([np.full([len(attention_to_boundary_inv), 1], "outside"),\
                                                                                                                 #attention_to_boundary_inv], axis=1), columns=["regiontype", "attention"])
                        attention_to_boundary.to_csv(tsv_file, sep="\t", index=False, header=False, mode="a")
                        #attention_to_boundary_inv.to_csv(tsv_file, sep="\t", index=False, header=False, mode="a")
                            
                    
        #for j in range(annotations.shape[1]):
        #    annotation_matrix[j] = annotation_matrix[j] / annotation_count[j]
        #    annotation_matrix_negative[j] = annotation_matrix_negative[j] / annotation_count_negative[j]
    return

def analyze_regionboundary(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)
    
    regionboundaries = [[1,4],[0,4],[3,4],[0,3],[1,3]]
    if not args.region_boundaries == "":
        lines = re.findall("[0-9]+[,-][0-9]+", args.region_boundaries)
        for regions in lines:
            regions = regions.split(",")
            regions = list(map(int, regions))
            regions = list(np.array(structure_and_head))
            regionboundaries.append(regions)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        
        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12
        
        positional_effects = torch.ones(num_layer, num_head, args.max_seq_length)
        if args.correct_position_bias:
            positional_effects_dir = os.path.join(args.positional_effects_dir, "positional_effects.npy")
            if os.path.isfile(positional_effects_dir):
                logger.info("positional_effects_dir: {}".format(positional_effects_dir))
                positional_effects = np.load(positional_effects_dir) / 100000
                positional_effects = np.where(positional_effects==0, np.min(positional_effects), positional_effects)
                positional_effects = torch.tensor(positional_effects)
        
        # count = 0
        regionboundary_matrix = np.zeros([len(regionboundaries), num_layer, num_head])
        regionboundary_count = np.zeros(len(regionboundaries))
        regionboundary_matrix_negative = np.zeros([len(regionboundaries), num_layer, num_head])
        regionboundary_count_negative = np.zeros(len(regionboundaries))
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
                # logger.info('outputs: {}'.format(len(outputs))) #
                # logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                attention = torch.stack(outputs[-1], 1)[:,:,:,:,:]
                # attention.shape = [batch_size, num_layer, num_head, token_length_after(query), token_length_before(key, value)]
                attention = torch.sum(attention, dim=-2)
                # attention.shape = [batch_size, num_layer, num_head, token_length_before(key, value)]
                #attention = torch.sum(attention, dim=-2)
                # attention.shape = [batch_size, num_layer, num_head, token_length_before(key, value)]
                #if num_layer==1:
                #    attention = attention[:, args.vis_layer-1:args.vis_layer, :, :, :]
                #    if num_head==1:
                #        attention = attention[:,:,args.vis_head-1:args.vis_head,:,:]

                # attention.shape = [batch_size, num_layer, num_head, token_length_after]
                # attention[:,:,:, :(kmer-1),:] = 0.0
                # attention[:,:,:, :,:(kmer-1)] = 0.0
                # attention[:,:,:,-(kmer-1):,:] = 0.0
                # attention[:,:,:,:,-(kmer-1):] = 0.0

                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                annotations = get_annotations(args, kmer, num_start, num_end, binary=False, tokenize=True, add_nonannotated=True)
                
                
                annotation_boundaries = np.zeros([annotations.shape[0], len(regionboundaries), annotations.shape[-1]])
                for boundary_num, regionboundary in enumerate(regionboundaries):
                    annotation_boundary = annotations[:, regionboundary[0], :]
                    annotation_boundary2 = annotations[:, regionboundary[1], :]
                    annotation_boundary = np.where(annotation_boundary==kmer, 0, annotation_boundary)
                    annotation_boundary2 = np.where(annotation_boundary2==kmer, 0, annotation_boundary2)
                    annotation_boundary = annotation_boundary * annotation_boundary2
                    annotation_boundary = np.where(annotation_boundary>0, 1, 0)
                    annotation_boundaries[:, boundary_num, :] = annotation_boundary
                annotation_boundaries_inv = annotation_boundaries * -1 + 1
                annotation_boundaries_inv[:,:,0] = 0
                annotation_boundaries_inv[:,:,-1] = 0
                
                # annotation_boundary.shape : batch, token_length
                
                
                # annotations = np.zeros([annot.shape[0], annot.shape[1], annot.shape[-1]-kmer+3])
                # for i in range(kmer):
                #    annotations[:, :, 1:-1] = annotations[:, :, 1:-1] + annot[:, :, i:i+annot.shape[-1]-kmer+1]
                
                # logger.info('attention_map: {}'.format(attention.shape))
                # logger.info('annotations: {}'.format(annotations.shape))
                '''
                for i, attention_sample in enumerate(attention):
                    annotation_all_matrix = annotation_all_matrix + torch.sum(attention_sample, dim=-1).cpu().detach().numpy()
                    for layer in range(num_layer):
                        for head in range(num_head):
                            annotation_matrix[layer, head] = annotation_matrix[layer, head] +\
                                                             np.sum(annotations[i] * attention_sample[layer, head].cpu().detach().numpy())
                            #if i==1 and layer==2 and head==0:
                            #    logger.info('annotation_all_matrix: {}, annotation_matrix: {}'.format(\
                            #                 attention_sample.cpu().detach().numpy()[layer,head],\
                            #                 annotations[i]))
                '''
                #regionboundary_count[0] += np.sum(annotation_boundary)
                #regionboundary_count_negative[0] += np.sum(annotation_boundary_inv)
                
                for boundary_num in range(len(regionboundaries)):
                    annotation_boundary, annotation_boundary_inv = annotation_boundaries[:,boundary_num,:], annotation_boundaries_inv[:,boundary_num,:]
                    regionboundary_count[boundary_num] += np.sum(annotation_boundary)
                    regionboundary_count_negative[boundary_num] += np.sum(annotation_boundary_inv)
                    annotation_boundary = torch.tensor(annotation_boundary).repeat(num_layer,num_head,1,1).transpose(0,1).transpose(0,2)
                    annotation_boundary = annotation_boundary / positional_effects.repeat(annotation_boundary.shape[0], 1, 1, 1)
                    annotation_boundary_inv = torch.tensor(annotation_boundary_inv).repeat(num_layer,num_head,1,1).transpose(0,1).transpose(0,2)
                    annotation_boundary_inv = annotation_boundary_inv / positional_effects.repeat(annotation_boundary_inv.shape[0], 1, 1, 1)

                    attention_to_boundary = torch.sum(torch.sum(attention * annotation_boundary, dim=0), dim=-1)
                    attention_to_others = torch.sum(torch.sum(attention * annotation_boundary_inv, dim=0), dim=-1)
                    # attention_to_boundary.shape: batch_size, num_layer, num_head, token_length_before(key, value)
                    # attention_from_boundary = torch.sum(torch.sum(attentions[:, layer, head, :, :], dim=-2) * annotation_boundary, dim=0)
                    regionboundary_matrix[boundary_num, :, :] = regionboundary_matrix[boundary_num, :, :] + attention_to_boundary.cpu().detach().numpy()
                    regionboundary_matrix_negative[boundary_num, :, :] = regionboundary_matrix_negative[boundary_num, :, :] + attention_to_others.cpu().detach().numpy()
                    
        #for j in range(annotations.shape[1]):
        #    annotation_matrix[j] = annotation_matrix[j] / annotation_count[j]
        #    annotation_matrix_negative[j] = annotation_matrix_negative[j] / annotation_count_negative[j]
    return regionboundary_matrix, regionboundary_matrix_negative, regionboundary_count, regionboundary_count_negative, regionboundaries

def analyze_regiontypes_specific(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)
    
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        
        regions_and_heads = []
        if args.specific_heads:
            lines = re.findall("[1-9][,-][0-9]+[,-][0-9]+", args.specific_heads)
            for region_and_head in lines:
                region_and_head = region_and_head.split(",")
                region_and_head = list(map(int, region_and_head))
                regions_and_heads.append(region_and_head)
        else:
            textfile = os.path.join(pred_output_dir, "analyze_regiontype.txt")
            regions_and_heads = []
            regiontypes = ("5'UTR", "3'UTR", "exon", "intron", "CDS")
            index_regiontypes = {"5'UTR":1, "3'UTR":2, "exon":3, "intron":4, "CDS":5}
            with open(textfile, 'r') as f:
                for line in f:
                    region_and_head = []
                    if re.match("region_type: [0-9a-zA-Z\']+, max_head: [0-9\-]+", line):
                        for regiontype in regiontypes:
                            if regiontype in line:
                                region_and_head.append(index_regiontypes[regiontype])
                                max_head = re.findall("max_head: [0-9\-]+", line)[0]
                                max_head = [int(i) for i in re.findall("[0-9]+", max_head)]
                                region_and_head.extend(max_head)
                                regions_and_heads.append(region_and_head)
        
        tsv_file_original = os.path.join(pred_output_dir, "analyze_regiontype_specific_type{}_layer{}_head{}.tsv")
        for region_and_head in regions_and_heads:
            tsv_file = tsv_file_original.format(region_and_head[0], region_and_head[1], region_and_head[2])
            with open(tsv_file, "w") as f:
                f.write("regiontype\tattention\n")
        logger.info("regions_and_heads: {}".format(regions_and_heads))
        
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        
        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12

        #if args.vis_layer > 0:
        #    num_layer = 1
        #    if args.vis_head > 0:
        #        num_head = 1
        
        #probe_matrix = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_matrix.npy')), device=args.device)
        #probe_beta = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_beta.npy')), device=args.device)
        #optimizer = Adam([probe_matrix], lr=probe_lr)
        
        #import matplotlib.pyplot as plt
        #import seaborn as sns
        #fig, ax = plt.subplots(1,2,figsize=(16,6))
        
        # count = 0        
        #regiontype_vs_attention = np.zeros([len(pred_dataset), 1, 2])
        #if args.region_type == 0:
        #    regiontype_vs_attention = np.zeros([len(pred_dataset), 5, 2])
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
                # logger.info('outputs: {}'.format(len(outputs))) #
                # logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                attention = torch.stack(outputs[-1], 1)[:,:,:,:,:]
                #if num_layer==1:
                #    attention = attention[:, args.vis_layer-1:args.vis_layer, :, :, :]
                #    if num_head==1:
                #        attention = attention[:,:,args.vis_head-1:args.vis_head,:,:]
                attention = torch.sum(attention, dim=-2)
                #logger.info("attention.shape {}".format(attention.shape))
                # attention.shape = [batch_size, 12, 12, token_length_before(key, value)]

                # attention.shape = [batch_size, num_layer, num_head, token_length_before]
                # attention[:,:,:, :(kmer-1),:] = 0.0
                # attention[:,:,:, :,:(kmer-1)] = 0.0
                # attention[:,:,:,-(kmer-1):,:] = 0.0
                # attention[:,:,:,:,-(kmer-1):] = 0.0

                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                annotations = get_annotations(args, kmer, num_start, num_end, binary=True, tokenize=False)
                
                # logger.info('attention_map: {}'.format(attention.shape))
                # logger.info('annotations: {}'.format(annotations.shape))
                # annotations.shape : batch, token_length
                '''
                for i, attention_sample in enumerate(attention):
                    annotation_all_matrix = annotation_all_matrix + torch.sum(attention_sample, dim=-1).cpu().detach().numpy()
                    for layer in range(num_layer):
                        for head in range(num_head):
                            annotation_matrix[layer, head] = annotation_matrix[layer, head] +\
                                                             np.sum(annotations[i] * attention_sample[layer, head].cpu().detach().numpy())
                            #if i==1 and layer==2 and head==0:
                            #    logger.info('annotation_all_matrix: {}, annotation_matrix: {}'.format(\
                            #                 attention_sample.cpu().detach().numpy()[layer,head],\
                            #                 annotations[i]))
                '''
                for i, attention_sample in enumerate(attention):
                    attention_sample_cls = attention_sample[:,:,0]
                    # attention_sample_cls.shape = [12, 12]
                    #attention_sample_cls = attention_sample_cls / torch.sum(attention_sample[:,:,:], dim=-1)
                    #i2 = index*batch_size + i
                    for region_and_head in regions_and_heads:
                        tsv_file = tsv_file_original.format(region_and_head[0], region_and_head[1], region_and_head[2])
                        new_line = ("{}\t{}\n".format(int(np.sum(annotations[i,region_and_head[0]-1])),\
                                                      float(attention_sample_cls[region_and_head[1]-1, region_and_head[2]-1,].cpu().detach().numpy())))
                        with open(tsv_file, 'a') as f:
                            f.write(new_line)
                 
                
                    # for j in range(annotations.shape[1]):
                    #    regiontype_vs_attention[i2,j,0] = np.sum(annotations[i,j])
                    #    regiontype_vs_attention[i2,j,1] = attention_sample_cls.cpu().detach().numpy()
                #logger.info("check: {}".format(regiontype_vs_attention[0]))
                
        #for j in range(annotations.shape[1]):
        #    annotation_matrix[j] = annotation_matrix[j] / annotation_count[j]
        #    annotation_matrix_negative[j] = annotation_matrix_negative[j] / annotation_count_negative[j]
        # np.save(os.path.join(args.predict_dir, 'analyze_annotation_negative_type{}.npy'.format(args.region_type)), annotation_matrix_negative)
        # np.save(os.path.join(args.predict_dir, 'analyze_annotation_count_type{}.npy'.format(args.region_type)), annotation_count)
        # np.save(os.path.join(args.predict_dir, 'analyze_annotation_count_negative_type{}.npy'.format(args.region_type)), annotation_count_negative)
    #return regiontype_vs_attention
    return

def analyze_regiontypes(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        
        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12

        #if args.vis_layer > 0:
        #    num_layer = 1
        #    if args.vis_head > 0:
        #        num_head = 1
        
        #probe_matrix = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_matrix.npy')), device=args.device)
        #probe_beta = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_beta.npy')), device=args.device)
        #optimizer = Adam([probe_matrix], lr=probe_lr)
        
        #import matplotlib.pyplot as plt
        #import seaborn as sns
        #fig, ax = plt.subplots(1,2,figsize=(16,6))
        
        # count = 0
        regiontype_matrix = np.zeros([1, num_layer, num_head])
        regiontype_count = np.zeros(1)
        regiontype_matrix_negative = np.zeros([1, num_layer, num_head])
        regiontype_count_negative = np.zeros(1)
        if args.region_type == 0:
            regiontype_matrix = np.zeros([5, num_layer, num_head])
            regiontype_count = np.zeros(5)
            regiontype_matrix_negative = np.zeros([5, num_layer, num_head])
            regiontype_count_negative = np.zeros(5)
        
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
                # logger.info('outputs: {}'.format(len(outputs))) #
                # logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                attention = torch.stack(outputs[-1], 1)[:,:,:,:,:]
                # attention.shape = [batch_size, num_layer, num_head, token_length_after(query), token_length_before(key, value)]
                attention = torch.sum(attention, dim=-2)
                # attention.shape = [batch_size, num_layer, num_head, token_length_before(key, value)]
                #if num_layer==1:
                #    attention = attention[:, args.vis_layer-1:args.vis_layer, :, :, :]
                #    if num_head==1:
                #        attention = attention[:,:,args.vis_head-1:args.vis_head,:,:]

                # attention.shape = [batch_size, num_layer, num_head, token_length_after]
                # attention[:,:,:, :(kmer-1),:] = 0.0
                # attention[:,:,:, :,:(kmer-1)] = 0.0
                # attention[:,:,:,-(kmer-1):,:] = 0.0
                # attention[:,:,:,:,-(kmer-1):] = 0.0

                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                annotations = get_annotations(args, kmer, num_start, num_end, binary=True, tokenize=False)
                
                # logger.info('attention_map: {}'.format(attention.shape))
                # logger.info('annotations: {}'.format(annotations.shape))
                '''
                for i, attention_sample in enumerate(attention):
                    annotation_all_matrix = annotation_all_matrix + torch.sum(attention_sample, dim=-1).cpu().detach().numpy()
                    for layer in range(num_layer):
                        for head in range(num_head):
                            annotation_matrix[layer, head] = annotation_matrix[layer, head] +\
                                                             np.sum(annotations[i] * attention_sample[layer, head].cpu().detach().numpy())
                            #if i==1 and layer==2 and head==0:
                            #    logger.info('annotation_all_matrix: {}, annotation_matrix: {}'.format(\
                            #                 attention_sample.cpu().detach().numpy()[layer,head],\
                            #                 annotations[i]))
                '''
                for i, attention_sample in enumerate(attention):
                    attention_sample_cls = attention_sample[:,:,0]
                    # attention_sample.shape = (num_layer, num_head, token_length_before(key, value))
                    # attention_sample_cls.shape = (num_layer, num_head)
                    #if i<5:
                    #    logger.info("torch.sum(attention_sample, dim=-1): {}".format(torch.sum(attention_sample, dim=-1)))
                    # attention_sample_cls = attention_sample_cls / torch.sum(attention_sample[:,:,:], dim=-1)
                    for j in range(annotations.shape[1]):
                        if np.sum(annotations[i,j]) > 0:
                            regiontype_matrix[j] = regiontype_matrix[j] + attention_sample_cls.cpu().detach().numpy()
                            regiontype_count[j] += 1
                            # if j==0:
                            #    logger.info("np.sum(annotations[i,j]): {}".format(np.sum(annotations[i,j])))
                        else:
                            regiontype_matrix_negative[j] = regiontype_matrix_negative[j] + attention_sample_cls.cpu().detach().numpy()
                            regiontype_count_negative[j] += 1
        #for j in range(annotations.shape[1]):
        #    annotation_matrix[j] = annotation_matrix[j] / annotation_count[j]
        #    annotation_matrix_negative[j] = annotation_matrix_negative[j] / annotation_count_negative[j]
        
    return regiontype_matrix, regiontype_matrix_negative, regiontype_count, regiontype_count_negative

def get_gccontent(args, num_start=0, num_end=-1, evaluate=True):
    gc_list = []
    
    import pandas as pd
    df = pd.read_csv(os.path.join(args.data_dir, 'dev.tsv'), sep='\t')
    if not evaluate:
        df = pd.read_csv(os.path.join(args.data_dir, 'train.tsv'), sep='\t')
    
    df = df[num_start:num_end]
    
    for i in df['sequence']:
        num_gc = i.count('C') + i.count('G')
        gc_list.append(num_gc/len(i))
    
    return gc_list

def analyze_gccontent_specific(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        
        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12

        if args.vis_layer > 0:
            num_layer = 1
            if args.vis_head > 0:
                num_head = 1
        
        #probe_matrix = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_matrix.npy')), device=args.device)
        #probe_beta = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_beta.npy')), device=args.device)
        #optimizer = Adam([probe_matrix], lr=probe_lr)
        
        #import matplotlib.pyplot as plt
        #import seaborn as sns
        #fig, ax = plt.subplots(1,2,figsize=(16,6))
        
        #count = 0
        #gc_low_matrix = np.zeros([num_layer, num_head])
        #gc_high_matrix = np.zeros([num_layer, num_head])
        #gc_high_count = 0
        #gc_low_count = 0
        gc_vs_attention = np.zeros([len(pred_dataset), 2])
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
                logger.info('outputs: {}'.format(len(outputs))) #
                logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                attention = torch.stack(outputs[-1], 1)[:,:,:,:,:]
                attention = torch.sum(attention, dim=-2)
                if num_layer==1:
                    attention = attention[:, args.vis_layer-1:args.vis_layer, :, :]
                    if num_head==1:
                        attention = attention[:,:,args.vis_head-1:args.vis_head,:]

                # attention.shape = [batch_size, num_layer, num_head, token_length_after, token_length_before]
                # attention[:,:,:, :(kmer-1),:] = 0.0
                # attention[:,:,:, :,:(kmer-1)] = 0.0
                # attention[:,:,:,-(kmer-1):,:] = 0.0
                # attention[:,:,:,:,-(kmer-1):] = 0.0

                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                gc_list = get_gccontent(args, evaluate=True)
                logger.info('attention_map: {}'.format(attention.shape))
                logger.info('gc_list: {}'.format(len(gc_list)))
                
                for i, attention_sample in enumerate(attention):
                    attention_sample_cls = torch.sum(attention_sample[:,:,0]).cpu().detach().numpy()
                    attention_sample_cls = attention_sample_cls / torch.sum(torch.sum(attention_sample[:,:,:])).cpu().detach().numpy()
                    j = index*batch_size + i
                    gc_vs_attention[j, 0] = gc_list[i]
                    gc_vs_attention[j, 1] = attention_sample_cls
                    
                
                
                #for i, attention_sample in enumerate(attention):
                #    attention_sample = torch.sum(attention_sample[:,:,0,:], dim=-1)
                #    if gc_list[i] > args.analyze_GCcontent_thresh:
                #            gc_high_matrix = gc_high_matrix + attention_sample.cpu().detach().numpy()
                #            gc_high_count += 1
                #    else:
                #            gc_low_matrix = gc_low_matrix + attention_sample.cpu().detach().numpy()
                #            gc_low_count += 1
                
    return gc_vs_attention

def analyze_gccontent(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        
        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12

        #if args.vis_layer > 0:
        #    num_layer = 1
        #    if args.vis_head > 0:
        #        num_head = 1
        
        #probe_matrix = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_matrix.npy')), device=args.device)
        #probe_beta = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_beta.npy')), device=args.device)
        #optimizer = Adam([probe_matrix], lr=probe_lr)
        
        #import matplotlib.pyplot as plt
        #import seaborn as sns
        #fig, ax = plt.subplots(1,2,figsize=(16,6))
        
        #count = 0
        gc_low_matrix = np.zeros([num_layer, num_head])
        gc_high_matrix = np.zeros([num_layer, num_head])
        gc_high_count = 0
        gc_low_count = 0
        gc_list_all = []
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
                logger.info('outputs: {}'.format(len(outputs))) #
                logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                attention = torch.stack(outputs[-1], 1)[:,:,:,:,:]
                attention = torch.sum(attention, dim=-2)
                #if num_layer==1:
                #    attention = attention[:, args.vis_layer-1:args.vis_layer, :, :, :]
                #    if num_head==1:
                #        attention = attention[:,:,args.vis_head-1:args.vis_head,:,:]

                # attention.shape = [batch_size, num_layer, num_head, token_length_after, token_length_before]
                # attention[:,:,:, :(kmer-1),:] = 0.0
                # attention[:,:,:, :,:(kmer-1)] = 0.0
                # attention[:,:,:,-(kmer-1):,:] = 0.0
                # attention[:,:,:,:,-(kmer-1):] = 0.0

                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                gc_list = get_gccontent(args, num_start, num_end, evaluate=True)
                gc_list_all.extend(gc_list)
                logger.info('attention_map: {}'.format(attention.shape))
                logger.info('gc_list: {}'.format(len(gc_list)))
                for i, attention_sample in enumerate(attention):
                    attention_sample_cls = attention_sample[:,:,0]
                    attention_sample_cls = attention_sample_cls / torch.sum(attention_sample[:,:,:], dim=-1)
                    if gc_list[i] > args.analyze_GCcontent_thresh:
                            gc_high_matrix = gc_high_matrix + attention_sample_cls.cpu().detach().numpy()
                            gc_high_count += 1
                    else:
                            gc_low_matrix = gc_low_matrix + attention_sample_cls.cpu().detach().numpy()
                            gc_low_count += 1
                            
    return gc_low_matrix, gc_low_count, gc_high_matrix, gc_high_count

def get_bpp_matrix(args, kmer, num_start, num_end, tokenize=False, training=False):
    # -2 includes [CLS] and [SEP] tokens
    nt_length = args.max_seq_length -2 + (kmer-1)
    bpp_shape = [num_end - num_start, nt_length, nt_length]
    path_tmp = args.data_dir
    
    if args.path_to_fasta:
        create_fasta_from_tsv(args.data_dir, num_start, num_end, args.path_to_fasta, training=training)
        path_tmp = args.path_to_fasta
    else:
        create_fasta_from_tsv(path_tmp, num_start, num_end, training=training)
    
    execute_linearpartition(path_tmp, args.path_to_linearpartition, training=training)
    bpp_matrix = get_base_pairing_prob(path_tmp, bpp_shape, training=training)
    #bpp_matrix.shape = [batch_size, nt_length, nt_length]
    
    if tokenize:
        bpp_matrix = tokenize_base_pairing_prob(bpp_matrix, kmer)
    
    return torch.tensor(bpp_matrix, dtype=torch.float)

def probe_structure_eval(args, model, tokenizer, kmer, probe_lr, probe_matrix_depth, probe_lambda, prefix="", output_example=-1):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        
        #evaluate = False if args.probe_train else True
        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        probe_matrix = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_matrix_layer{}.npy'.format(args.vis_layer))))
        #optimizer = Adam([probe_matrix], lr=probe_lr)
        
        
        
        loss_probe = torch.zeros(1)
        count = 0
        for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            num_layer = args.vis_layer
            #num_head = args.vis_head

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                #logger.info('outputs: {}'.format(len(outputs))) #3
                #logger.info('outputs[2]: {}'.format(len(outputs[2]))) #13(initial embedding + 12layer)
                embedded_features = outputs[2][num_layer][:, 1:-1, :]
                #embedded_features.shape = [batch_size, token_length-2, hidden_dimensions(should be 768)]
                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                logger.info('num_start: {}, num_end: {}, len(pred_dataset): {}'.format(num_start, num_end, len(pred_dataset)))
                tr = False if evaluate else True
                bpp_matrix = get_bpp_matrix(args, kmer, num_start, num_end, tokenize=True, training=tr)
                logger.info('eval embedded_features: {}'.format(embedded_features.shape))
                logger.info('eval bpp_matrix: {}'.format(bpp_matrix.shape))

                for i, l in enumerate(embedded_features):
                    # l.shape = [token_length-2, hidden_dimensions]
                    tmp_index = np.arange(l.shape[0])
                    xx, yy = np.meshgrid(tmp_index, tmp_index)
                    l2 = torch.matmul(probe_matrix, torch.transpose((l[yy]-l[xx]), 1,2))
                    # l2.shape = [token_length-2, probe_B_hidden_size, token_length-2]
                    l3 = torch.transpose(l2, 0,1).reshape(probe_matrix.shape[0], l.shape[0]*l.shape[0])
                    # l3.shape = [probe_B_hidden_size, (token_length-2)**2]
                    dist_matrix = torch.diag(torch.matmul(l3.T,l3)).reshape(l.shape[0],l.shape[0])
                    '''
                    mask = torch.ones(dist_matrix.shape)
                    mask -= torch.diag(torch.ones(dist_matrix.shape[0]), diagonal=0)
                    mask_width = 5
                    for j in range(1, mask_width):
                        mask -= torch.diag(torch.ones([dist_matrix.shape[0]-j]), diagonal=j)
                        mask -= torch.diag(torch.ones([dist_matrix.shape[0]-j]), diagonal=-j)
                    dist_matrix = dist_matrix*mask
                    loss_probe = (torch.sum(abs(bpp_matrix[i]-dist_matrix)) + probe_lambda*torch.sum(abs(bpp_matrix[i])))/(l.shape[0]**2)
                    '''
                    loss_probe = loss_probe + (torch.sum(abs(bpp_matrix[i]-dist_matrix)) + probe_lambda*torch.sum(abs(bpp_matrix[i])))/(l.shape[0]**2)
                    count += 1
                    
                    if index*batch_size + i + 1 == output_example:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        fig, ax = plt.subplots(1,2,figsize=(16,6))
                        dist_matrix = dist_matrix.detach().numpy()
                        sns.heatmap(bpp_matrix[i], ax=ax[0])
                        sns.heatmap(dist_matrix, ax=ax[1])
                        ax[0].set_title('base pairing probability matrix: sample{}'.format(output_example))
                        ax[1].set_title('predicted matrix: sample{}'.format(output_example))
                        fig.savefig(os.path.join(args.predict_dir, 'structural_probe_sample{}.png'.format(epoch, output_example)))
                        plt.clf()
                        plt.close()

    logger.info("***** probe evaluation loss *****".format(prefix))
    logger.info("avg probe eval loss: {}".format(loss_probe/count))
    return loss_probe / count

def probe_structure(args, model, tokenizer, kmer, probe_lr, probe_matrix_depth, probe_lambda, prefix="", output_example=-1):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    with open(os.path.join(args.predict_dir, 'eval_probe.txt'), 'a') as f:
        f.write('> probe_lr: {}, probe_matrix_depth: {}, probe_lambda: {}, epoch: {}\n'.format(probe_lr, probe_matrix_depth, probe_lambda, args.probe_epoch))
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        
        evaluate = False #if args.probe_train else True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])

        probe_matrix = torch.tensor(np.random.uniform(0, 1, (probe_matrix_depth, 768)), dtype=torch.float, requires_grad=True)
        optimizer = Adam([probe_matrix], lr=probe_lr)

        #import matplotlib.pyplot as plt
        #import seaborn as sns
        #fig, ax = plt.subplots(1,2,figsize=(16,6))
        for epoch in range(int(args.probe_epoch)):
            for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                num_layer = args.vis_layer
                #num_head = args.vis_head

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in TOKEN_ID_GROUP else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    outputs = model(**inputs)
                    #logger.info('outputs: {}'.format(len(outputs))) #3
                    #logger.info('outputs[2]: {}'.format(len(outputs[2]))) #13(initial embedding + 12layer)
                    embedded_features = outputs[2][num_layer][:, 1:-1, :]
                    #embedded_features.shape = [batch_size, token_length-2, hidden_dimensions(should be 768)]
                    num_start = index * batch_size
                    num_end = (index+1) * batch_size
                    if num_end > len(pred_dataset):
                        num_end = len(pred_dataset)
                    tr = False if evaluate else True
                    bpp_matrix = get_bpp_matrix(args, kmer, num_start, num_end, tokenize=True, training=tr)
                    logger.info('embedded_features: {}'.format(embedded_features.shape))
                    logger.info('bpp_matrix: {}'.format(bpp_matrix.shape))
                    
                    torch.set_grad_enabled(True)
                    loss_probe = torch.zeros(1)
                    for i, l in enumerate(embedded_features):
                        # l.shape = [token_length-2, hidden_dimensions]
                        tmp_index = np.arange(l.shape[0])
                        xx, yy = np.meshgrid(tmp_index, tmp_index)
                        l2 = torch.matmul(probe_matrix, torch.transpose((l[yy]-l[xx]), 1,2))
                        # l2.shape = [token_length-2, probe_B_hidden_size, token_length-2]
                        l3 = torch.transpose(l2, 0,1).reshape(probe_matrix.shape[0], l.shape[0]*l.shape[0])
                        # l3.shape = [probe_B_hidden_size, (token_length-2)**2]
                        dist_matrix = torch.diag(torch.matmul(l3.T,l3)).reshape(l.shape[0],l.shape[0])
                        '''
                        mask = torch.ones(dist_matrix.shape)
                        mask -= torch.diag(torch.ones(dist_matrix.shape[0]), diagonal=0)
                        mask_width = 5
                        for j in range(1, mask_width):
                            mask -= torch.diag(torch.ones([dist_matrix.shape[0]-j]), diagonal=j)
                            mask -= torch.diag(torch.ones([dist_matrix.shape[0]-j]), diagonal=-j)
                        dist_matrix = dist_matrix*mask
                        loss_probe = (torch.sum(abs(bpp_matrix[i]-dist_matrix)) + probe_lambda*torch.sum(abs(bpp_matrix[i])))/(l.shape[0]**2)
                        '''
                        loss_probe = (torch.sum(abs(bpp_matrix[i]-dist_matrix)) + probe_lambda*torch.sum(abs(bpp_matrix[i])))/(l.shape[0]**2)
                        optimizer.zero_grad()
                        loss_probe.backward()
                        optimizer.step()
                        if (index*batch_size + i)%args.logging_steps==0:
                            np.save(os.path.join(args.probe_matrix_dir, 'probe_matrix_layer{}.npy'.format(args.vis_layer)), probe_matrix.detach().numpy())
                            eval_loss = probe_structure_eval(args, model, tokenizer, kmer, probe_lr,\
                                                             probe_matrix_depth, probe_lambda, prefix, output_example)
                            with open(os.path.join(args.predict_dir, 'eval_probe.txt'), 'a') as f:
                                f.write('eval_loss: {}\n'.format(float(eval_loss)))
    np.save(os.path.join(args.probe_matrix_dir, 'probe_matrix_layer{}.npy'.format(args.vis_layer)), probe_matrix.detach().numpy())
    eval_loss = probe_structure_eval(args, model, tokenizer, kmer, probe_lr,\
                                     probe_matrix_depth, probe_lambda, prefix, output_example)
    with open(os.path.join(args.predict_dir, 'eval_probe.txt'), 'a') as f:
        f.write('final_eval_loss: {}\n'.format(float(eval_loss)))
    return probe_matrix


def probe_attention_eval(args, model, tokenizer, kmer, probe_lr, probe_matrix_depth, probe_lambda, eval_count=0, prefix="", output_example=-1):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        
        evaluate = True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12

        if args.vis_layer > 0:
            num_layer = 1
            if args.vis_head > 0:
                num_head = 1
        
        probe_matrix = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_matrix.npy')), device=args.device)
        probe_beta = torch.tensor(np.load(os.path.join(args.probe_matrix_dir, 'probe_beta.npy')), device=args.device)
        #optimizer = Adam([probe_matrix], lr=probe_lr)
        
        #import matplotlib.pyplot as plt
        #import seaborn as sns
        #fig, ax = plt.subplots(1,2,figsize=(16,6))
        
        loss_probe = torch.zeros(1, device=args.device)
        count = 0
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
                logger.info('outputs: {}'.format(len(outputs))) #
                logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                attention = torch.stack(outputs[-1], 1)[:,:,:,1:-1,1:-1]
                if num_layer==1:
                    attention = attention[:, args.vis_layer-1:args.vis_layer, :, :, :]
                    if num_head==1:
                        attention = attention[:,:,args.vis_head-1:args.vis_head,:,:]

                # attention.shape = [batch_size, num_layer, num_head, token_length_after, token_length_before]
                # attention[:,:,:, :(kmer-1),:] = 0.0
                # attention[:,:,:, :,:(kmer-1)] = 0.0
                # attention[:,:,:,-(kmer-1):,:] = 0.0
                # attention[:,:,:,:,-(kmer-1):] = 0.0

                num_start = index * batch_size
                num_end = (index+1) * batch_size
                if num_end > len(pred_dataset):
                    num_end = len(pred_dataset)
                tr = False if evaluate else True
                bpp_matrix = get_bpp_matrix(args, kmer, num_start, num_end, tokenize=False, training=tr).to(args.device)
                logger.info('attention_map: {}'.format(attention.shape))
                logger.info('bpp_matrix: {}'.format(bpp_matrix.shape))

                
                loss_probe = torch.zeros(1)
                for i, attention_sample in enumerate(attention):
                    real_scores = torch.zeros([attention_sample.shape[0], attention_sample.shape[1],\
                                               attention_sample.shape[-1]+kmer-1, attention_sample.shape[-1]+kmer-1], device=args.device)
                    count_correction = torch.zeros(real_scores.shape, device=args.device)
                    #logger.info('attention_sample: {}'.format(attention_sample.shape))
                    #logger.info('real_scores: {}'.format(real_scores.shape))
                    for k1 in range(attention_sample.shape[-1]):
                        for k2 in range(attention_sample.shape[-1]):
                            #logger.info('real_scores[:,:,k1:k1+kmer, k2:k2+kmer]: {}, {}'.format(real_scores[:,:,k1:k1+kmer, k2:k2+kmer].shape, real_scores.device))
                            #logger.info('attention_sample[:,:,k1,k2]: {}, {}'.format(attention_sample[:,:,k1,k2].shape, attention_sample.device))
                            real_scores[:,:,k1:k1+kmer, k2:k2+kmer] = real_scores[:,:,k1:k1+kmer, k2:k2+kmer] + attention_sample[:,:,k1:k1+1,k2:k2+1]
                            count_correction[:,:,k1:k1+kmer, k2:k2+kmer] = count_correction[:,:,k1:k1+kmer, k2:k2+kmer] + 1.0
                    real_scores = real_scores/count_correction
                    for i_layer in range(num_layer):
                        for i_head in range(num_head):
                            norm = np.linalg.norm(real_scores[i_layer, i_head].cpu().detach().numpy())
                            real_scores[i_layer, i_head] = real_scores[i_layer, i_head]/torch.tensor(norm, device=args.device)
                    real_scores = real_scores + torch.transpose(real_scores, 2,3)
                    real_scores = real_scores / 2.0

                    #APC
                    prob_predict = torch.zeros(real_scores.shape[-2:], device=args.device)
                    for i_layer in range(num_layer):
                        for i_head in range(num_head):
                            real_scores[i_layer, i_head] = real_scores[i_layer, i_head] - \
                                                            torch.matmul(\
                                                            torch.sum(real_scores[i_layer,i_head], dim=1).reshape(real_scores.shape[2], 1),\
                                                            torch.sum(real_scores[i_layer,i_head], dim=0).reshape(1, real_scores.shape[3])\
                                                            )/torch.sum(real_scores[i_layer,i_head])
                    real_scores_vis = 0
                    if index*batch_size + i + 1 == output_example:
                        real_scores_vis = real_scores.clone()
                    #prob_predict = torch.sum(torch.sum(real_scores * probe_matrix, dim=0), dim=0)
                    prob_predict = torch.sum(torch.sum(real_scores * probe_matrix, dim=0), dim=0)
                    prob_predict = 1/(1 + torch.exp(- probe_beta - prob_predict))
                    '''
                    mask = torch.ones(prob_predict.shape, device=args.device)
                    mask = mask - torch.diag(torch.ones(mask.shape[0], device=args.device), diagonal=0)
                    mask_size = 5
                    for j in range(1, mask_size):
                        mask = mask - torch.diag(torch.ones([mask.shape[0]-j], device=args.device), diagonal=j)
                        mask = mask - torch.diag(torch.ones([mask.shape[0]-j], device=args.device), diagonal=-j)
                    prob_predict = prob_predict*mask
                    '''
                    loss_probe = loss_probe + torch.sum(abs(bpp_matrix[i] - prob_predict)) + probe_lambda*torch.sum(abs(probe_matrix))
                    count += 1
                    
                    if index*batch_size + i + 1 == output_example:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        fig, ax = plt.subplots(2,3,figsize=(24,12))
                        #fig, ax = plt.subplots(2,4,figsize=(32,12))
                        #num_vis_layer = 11
                        #num_vis_head = 10
                        a = torch.sum(torch.sum(attention_sample, dim=0), dim=0)
                        b = torch.sum(torch.sum(real_scores_vis, dim=0), dim=0)
                        c = torch.sum(torch.sum(probe_matrix, dim=0), dim=0)

                        sns.heatmap(bpp_matrix[i].cpu().detach().numpy(), ax=ax[0,0])
                        sns.heatmap(prob_predict.cpu().detach().numpy(), ax=ax[0,1])
                        #sns.heatmap(attention_sample[num_vis_layer, num_vis_head].cpu().detach().numpy(), ax=ax[0,2])
                        sns.heatmap(a.cpu().detach().numpy(), ax=ax[0,2])
                        #sns.heatmap(probe_matrix.cpu().detach().numpy(), ax=ax[1,0])
                        sns.heatmap(c.cpu().detach().numpy(), ax=ax[1,0])
                        #sns.heatmap(real_scores_vis[num_vis_layer, num_vis_head].cpu().detach().numpy(), ax=ax[1,1])
                        sns.heatmap(b.cpu().detach().numpy(), ax=ax[1,1])
                        #sns.heatmap(real_scores_vis2[num_vis_layer, num_vis_head].cpu().detach().numpy(), ax=ax[1,2])
                        ax[0,0].set_title('base pairing probability matrix: sample{}'.format(output_example))
                        ax[0,1].set_title('predicted matrix: sample{}'.format(output_example))
                        ax[0,2].set_title('original attention: sample{}'.format(output_example))
                        ax[1,0].set_title('probe matrix: sample{}'.format(output_example))
                        ax[1,1].set_title('modified original attention: sample{}'.format(output_example))
                        ax[1,2].set_title('modified original attention2: sample{}'.format(output_example))
                        fig.savefig(os.path.join(args.predict_dir, 'attention_probe_eval{}_sample{}.png'.format(eval_count, output_example)))
                        plt.clf()
                        plt.close()
    logger.info("***** probe evaluation loss *****".format(prefix))
    logger.info("avg probe eval loss: {}".format(loss_probe/count))
    return loss_probe / count


def probe_attention(args, model, tokenizer, kmer, probe_lr, probe_matrix_depth, probe_lambda, prefix="", output_example=-1):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    with open(os.path.join(args.predict_dir, 'eval_probe.txt'), 'a') as f:
        f.write('> probe_lr: {}, probe_matrix_depth: {}, probe_lambda: {}, epoch: {}\n'.format(probe_lr, probe_matrix_depth, probe_lambda, args.probe_epoch))
    
    eval_count = 0
    probe_lambda = torch.tensor(probe_lambda, dtype=torch.float, device=args.device)
    #import matplotlib.pyplot as plt
    #import seaborn as sns
    #fig, ax = plt.subplots(2,3,figsize=(24,12))
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
        
        evaluate = False # if args.probe_train else True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        #attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        num_layer = 12
        num_head = 12

        if args.vis_layer > 0:
            num_layer = 1
            if args.vis_head > 0:
                num_head = 1
        
        probe_matrix = torch.tensor(np.random.uniform(0, 1, (num_layer, num_head, args.max_seq_length -2 + kmer -1, args.max_seq_length -2 + kmer -1)), dtype=torch.float, device=args.device, requires_grad=True)
        probe_beta = torch.tensor(np.random.uniform(0,1,1), dtype=torch.float, device=args.device, requires_grad=True)
        optimizer = Adam([probe_matrix], lr=probe_lr)
        
        #attention_scores = np.zeros([len(pred_dataset), 12, 12, args.max_seq_length, args.max_seq_length])
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        for epoch in range(int(args.probe_epoch)):
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
                    logger.info('outputs: {}'.format(len(outputs))) #
                    logger.info('outputs[-1]: {}'.format(len(outputs[-1]))) #
                    attention = torch.stack(outputs[-1], 1)[:,:,:,1:-1,1:-1]
                    if num_layer==1:
                        attention = attention[:, args.vis_layer-1:args.vis_layer, :, :, :]
                        if num_head==1:
                            attention = attention[:,:,args.vis_head-1:args.vis_head,:,:]
                    
                    #attention.shape = [batch_size, num_layer, num_head, token_length_after, token_length_before]
                    #attention[:,:,:, :(kmer-1),:] = 0.0
                    #attention[:,:,:, :,:(kmer-1)] = 0.0
                    #attention[:,:,:,-(kmer-1):,:] = 0.0
                    #attention[:,:,:,:,-(kmer-1):] = 0.0
                    
                    
                    num_start = index * batch_size
                    num_end = (index+1) * batch_size
                    if num_end > len(pred_dataset):
                        num_end = len(pred_dataset)
                    tr = False if evaluate else True
                    bpp_matrix = get_bpp_matrix(args, kmer, num_start, num_end, tokenize=False, training=tr).to(args.device)
                    logger.info('attention_map: {}'.format(attention.shape))
                    logger.info('bpp_matrix: {}'.format(bpp_matrix.shape))
                    
                    torch.set_grad_enabled(True)
                    torch.autograd.set_detect_anomaly(True)
                    loss_probe = torch.zeros(1)
                    for i, attention_sample in enumerate(attention):
                        real_scores = torch.zeros([attention_sample.shape[0], attention_sample.shape[1],\
                                                   attention_sample.shape[-1]+kmer-1, attention_sample.shape[-1]+kmer-1], device=args.device)
                        count_correction = torch.zeros(real_scores.shape, device=args.device)
                        #logger.info('attention_sample: {}'.format(attention_sample.shape))
                        #logger.info('real_scores: {}'.format(real_scores.shape))
                        for k1 in range(attention_sample.shape[-1]):
                            for k2 in range(attention_sample.shape[-1]):
                                #logger.info('real_scores[:,:,k1:k1+kmer, k2:k2+kmer]: {}, {}'.format(real_scores[:,:,k1:k1+kmer, k2:k2+kmer].shape, real_scores.device))
                                #logger.info('attention_sample[:,:,k1,k2]: {}, {}'.format(attention_sample[:,:,k1,k2].shape, attention_sample.device))
                                real_scores[:,:,k1:k1+kmer, k2:k2+kmer] = real_scores[:,:,k1:k1+kmer, k2:k2+kmer] + attention_sample[:,:,k1:k1+1,k2:k2+1]
                                count_correction[:,:,k1:k1+kmer, k2:k2+kmer] = count_correction[:,:,k1:k1+kmer, k2:k2+kmer] + 1.0
                        real_scores = real_scores/count_correction
                        for i_layer in range(num_layer):
                            for i_head in range(num_head):
                                norm = np.linalg.norm(real_scores[i_layer, i_head].cpu().detach().numpy())
                                real_scores[i_layer, i_head] = real_scores[i_layer, i_head]/torch.tensor(norm, device=args.device)
                        real_scores = real_scores + torch.transpose(real_scores, 2,3)
                        real_scores = real_scores / 2.0
                        
                        #APC
                        prob_predict = torch.zeros(real_scores.shape[-2:], device=args.device)
                        for i_layer in range(num_layer):
                            for i_head in range(num_head):
                                real_scores[i_layer, i_head] = real_scores[i_layer, i_head] - \
                                                                torch.matmul(\
                                                                torch.sum(real_scores[i_layer,i_head], dim=1).reshape(real_scores.shape[2], 1),\
                                                                torch.sum(real_scores[i_layer,i_head], dim=0).reshape(1, real_scores.shape[3])\
                                                                )/torch.sum(real_scores[i_layer,i_head])
                        real_scores_vis = 0
                        if index*batch_size + i + 1 == output_example:
                            real_scores_vis = real_scores.clone()
                        #prob_predict = torch.sum(torch.sum(real_scores * probe_matrix, dim=0), dim=0)
                        prob_predict = torch.sum(torch.sum(real_scores * probe_matrix, dim=0), dim=0)
                        prob_predict = 1/(1 + torch.exp(- probe_beta - prob_predict))
                        '''
                        mask = torch.ones(prob_predict.shape, device=args.device)
                        mask = mask - torch.diag(torch.ones(mask.shape[0], device=args.device), diagonal=0)
                        mask_size = 5
                        for j in range(1, mask_size):
                            mask = mask - torch.diag(torch.ones([mask.shape[0]-j], device=args.device), diagonal=j)
                            mask = mask - torch.diag(torch.ones([mask.shape[0]-j], device=args.device), diagonal=-j)
                        prob_predict = prob_predict*mask
                        '''
                        #print(bpp_matrix.device)
                        #print(prob_predict.device)
                        #print(probe_lambda.device)
                        #print(probe_matrix.device)
                        loss_probe = torch.sum(abs(bpp_matrix[i] - prob_predict)) + probe_lambda*torch.sum(abs(probe_matrix))
                        optimizer.zero_grad()
                        loss_probe.backward()
                        optimizer.step()

                        if (index*batch_size + i)%args.logging_steps==0:
                            np.save(os.path.join(args.probe_matrix_dir, 'probe_matrix.npy'), probe_matrix.cpu().detach().numpy())
                            np.save(os.path.join(args.probe_matrix_dir, 'probe_beta.npy'), probe_beta.cpu().detach().numpy())
                            eval_loss = probe_attention_eval(args, model, tokenizer, kmer, probe_lr,\
                                                             probe_matrix_depth, probe_lambda, eval_count, prefix, output_example)
                            eval_count += 1
                            with open(os.path.join(args.predict_dir, 'eval_probe.txt'), 'a') as f:
                                f.write('eval_loss: {}\n'.format(float(eval_loss)))
    np.save(os.path.join(args.probe_matrix_dir, 'probe_matrix.npy'), probe_matrix.cpu().detach().numpy())
    np.save(os.path.join(args.probe_matrix_dir, 'probe_beta.npy'), probe_beta.cpu().detach().numpy())
    eval_loss = probe_structure_eval(args, model, tokenizer, kmer, probe_lr,\
                                     probe_matrix_depth, probe_lambda, eval_count, prefix, output_example)
    with open(os.path.join(args.predict_dir, 'eval_probe.txt'), 'a') as f:
        f.write('final_eval_loss: {}\n'.format(float(eval_loss)))
    return probe_matrix, probe_beta
'''
                        if (index*batch_size + i)%30==0:
                            logger.info('attention probe epoch: {}, step: {}, avg loss: {}'.format(epoch, i, loss_probe))
                        if index*batch_size + i + 1 == output_example and epoch%9 == 0:
                            if epoch==0:
                                import matplotlib.pyplot as plt
                                import seaborn as sns
                            fig, ax = plt.subplots(2,3,figsize=(24,12))
                            #fig, ax = plt.subplots(2,4,figsize=(32,12))
                            num_vis_layer = 11
                            num_vis_head = 10
                            a = torch.sum(torch.sum(attention_sample, dim=0), dim=0)
                            b = torch.sum(torch.sum(real_scores_vis, dim=0), dim=0)
                            c = torch.sum(torch.sum(probe_matrix, dim=0), dim=0)
                            
                            sns.heatmap(bpp_matrix[i].cpu().detach().numpy(), ax=ax[0,0])
                            sns.heatmap(prob_predict.cpu().detach().numpy(), ax=ax[0,1])
                            #sns.heatmap(attention_sample[num_vis_layer, num_vis_head].cpu().detach().numpy(), ax=ax[0,2])
                            sns.heatmap(a.cpu().detach().numpy(), ax=ax[0,2])
                            #sns.heatmap(probe_matrix.cpu().detach().numpy(), ax=ax[1,0])
                            sns.heatmap(c.cpu().detach().numpy(), ax=ax[1,0])
                            #sns.heatmap(real_scores_vis[num_vis_layer, num_vis_head].cpu().detach().numpy(), ax=ax[1,1])
                            sns.heatmap(b.cpu().detach().numpy(), ax=ax[1,1])
                            #sns.heatmap(real_scores_vis2[num_vis_layer, num_vis_head].cpu().detach().numpy(), ax=ax[1,2])
                            ax[0,0].set_title('base pairing probability matrix: sample{}'.format(output_example))
                            ax[0,1].set_title('predicted matrix: sample{}'.format(output_example))
                            ax[0,2].set_title('original attention: sample{}'.format(output_example))
                            ax[1,0].set_title('probe matrix: sample{}'.format(output_example))
                            ax[1,1].set_title('modified original attention: sample{}'.format(output_example))
                            ax[1,2].set_title('modified original attention2: sample{}'.format(output_example))
                            fig.savefig(os.path.join(args.predict_dir, 'attention_probe_epoch{}_sample{}.png'.format(epoch, output_example)))
                            plt.clf()
                            plt.close()
    return probe_matrix
'''

def analyze_attentionrollout(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)
    

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
            
            
        evaluate = False if args.visualize_train else True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
            
        #attention_scores = np.zeros([len(pred_dataset), args.max_seq_length])
        attention_scores = np.zeros([len(pred_dataset), 12, 12, 4])
        
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        #if args.vis_layer==0 and args.vis_head==0:
        #    all_attention_scores = np.zeros([200, 12, 12, args.max_seq_length, args.max_seq_length])
        
        for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            num_layer = args.vis_layer
            num_head = args.vis_head
            
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                #print(len(outputs))
                #print(len(outputs[-1]))
                #print('attention', attention.shape)
                
                attention = torch.stack(outputs[-1], 1)
                #attention.shape = torch.Size([batch_size, num_layer, num_head, token_length_after, token_length_before)
                #attention = torch.sum(attention, dim=2)
                #attention.shape = torch.Size([batch_size, num_layer, token_length_after, token_length_before)
                #attention_prev = attention[:,0,:,:]
                for layer in range(attention.shape[1]):
                    for head in range(attention.shape[1]):
                        attention_scores[index*batch_size:index*batch_size+len(batch[0]),layer,head,0] = torch.sum(attention[:,layer,head,0,1:-1], dim=-1).cpu().numpy()
                        attention_scores[index*batch_size:index*batch_size+len(batch[0]),layer,head,1] = torch.sum(attention[:,layer,head,1:-1,0], dim=-1).cpu().numpy()
                        attention_scores[index*batch_size:index*batch_size+len(batch[0]),layer,head,2] = torch.sum(attention[:,layer,head,0,:], dim=-1).cpu().numpy()
                        attention_scores[index*batch_size:index*batch_size+len(batch[0]),layer,head,3] = torch.sum(attention[:,layer,head,:,0], dim=-1).cpu().numpy()
                '''
                scale = 1
                if index < 2:
                    for layer in range(1, attention.shape[1]):
                        attention_next = attention[:,layer,:,:] #.transpose(-2,-1)
                        for before in range(attention_prev.shape[-1]):
                            attention_tmp = attention_prev[:,:,before]
                            attention_tmp = attention_tmp.reshape(attention_prev.shape[0]*attention_prev.shape[1],)
                            attention_tmp = attention_tmp.expand((attention_prev.shape[-1], attention_tmp.shape[0]))
                            attention_tmp = attention_tmp.transpose(0,1).reshape(attention_prev.shape[0],\
                                                                                 attention_prev.shape[1],\
                                                                                 attention_prev.shape[2]).transpose(-1,-2)
                            #tmp_size = attention_prev.shape[0]*attention_prev.shape[1]*attention_prev.shape[2]
                            attention_tmp = attention_tmp * attention_next
                            #attention_tmp = torch.min(attention_tmp, axis=0).values.view(attention_prev.shape[0],\
                            #                                                              attention_prev.shape[1],\
                            #                                                              attention_prev.shape[2])
                            attention_prev[:,:,before] = torch.sum(attention_tmp, dim=-1)
                        attention_scores[index*batch_size:index*batch_size+len(batch[0]),layer,0] = torch.sum(attention_prev[:,0,:], dim=-1).cpu().numpy()
                        attention_scores[index*batch_size:index*batch_size+len(batch[0]),layer,1] = torch.sum(attention_prev[:,:,0], dim=-1).cpu().numpy()
                '''
                
                '''
                    tmp_size = attention.shape[0]*attention.shape[2]*attention.shape[3]
                    for token_idx in range(attention.shape[-1]):
                        attention_keep = torch.zeros((attention.shape[0], attention.shape[2], attention.shape[3]))
                        for layer in range(attention.shape[1]):
                            if layer ==0:
                                attention_tmp = attention[:,layer,:,token_idx].reshape(attention.shape[0]*attention.shape[2],)
                                attention_tmp = attention_tmp.expand((attention.shape[-1], attention_tmp.shape[0]))
                                attention_tmp = attention_tmp.transpose(0,1).reshape(attention.shape[0],\
                                                                                     attention.shape[2],\
                                                                                     attention.shape[3]).transpose(-1,-2)
                                attention_tmp = torch.stack((attention_tmp.reshape(tmp_size,), \
                                                             attention[:,layer,token_idx,:].reshape(tmp_size,)))
                                attention_keep = torch.min(attention_tmp, axis=0).values.view(attention.shape[0],\
                                                                                              attention.shape[2],\
                                                                                              attention.shape[3])
                            else:
                                attention_keep = attention_keep.transpose(-1,-2)
                                attention_keep = torch.stack(attention_keep.reshape(tmp_size,),\
                                                             attention[:,layer,token_idx,:].reshape(tmp_size,))
                                attention_keep = torch.min(attention_tmp, axis=0).values.view(attention.shape[0],\
                                                                                              attention.shape[2],\
                                                                                              attention.shape[3])
                                aaa
                            aaa
                        aaa
                '''
                        
                        
                            
                #attention_prev.shape = torch.Size([batch_size, token_length_after, token_length_before)
                    
                #logger.info("attention.shape: {}".format(attention.shape))
                
                #attention = attention.cpu().numpy()[:, (kmer+1):-(kmer+1)]
                #attention_scores[index*batch_size:index*batch_size+len(batch[0]),:] = attention_prev[:,0,:].cpu().numpy()
        
        #attention_scores_save = np.zeros([len(pred_dataset), args.max_seq_length+kmer-3])
        
        #logger.info("saving file: attention_scores_save to atten.npy (%s)", attention_scores_save.shape)
            #logger.info("saving file: all_probs to pred_results.npy (%s)", all_probs.shape)
            #all_probs = all_probs/float(len(visualization_models))
        #np.save(os.path.join(args.predict_dir, "atten.npy"), attention_scores_save)
        logger.info("saving file: attention_scores to atten.npy (%s)", attention_scores.shape)
        np.save(os.path.join(args.predict_dir, "atten.npy"), attention_scores)
    return
    

def analyze_motif(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)
    

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
            
            
        evaluate = False if args.visualize_train else True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
            
        attention_scores = np.zeros([len(pred_dataset), args.max_seq_length])
        
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        #if args.vis_layer==0 and args.vis_head==0:
        #    all_attention_scores = np.zeros([200, 12, 12, args.max_seq_length, args.max_seq_length])
        
        for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            num_layer = args.vis_layer
            num_head = args.vis_head
            
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                #attention = outputs[-1][-1]
                #attention.shape = torch.Size([batch_size, num_head, token_length_after, token_length_before)
                #print(len(outputs))
                #print(len(outputs[-1]))
                #print('attention', attention.shape)
                
                attention = outputs[-1][-1][:, :, 0, :]
                attention = torch.sum(attention, dim=1)
                #logger.info("attention.shape: {}".format(attention.shape))
                
                #attention = attention.cpu().numpy()[:, (kmer+1):-(kmer+1)]
                attention_scores[index*batch_size:index*batch_size+len(batch[0]),:] = attention.cpu().numpy()
        
        #attention_scores_save = np.zeros([len(pred_dataset), args.max_seq_length+kmer-3])
        
        #logger.info("saving file: attention_scores_save to atten.npy (%s)", attention_scores_save.shape)
            #logger.info("saving file: all_probs to pred_results.npy (%s)", all_probs.shape)
            #all_probs = all_probs/float(len(visualization_models))
        #np.save(os.path.join(args.predict_dir, "atten.npy"), attention_scores_save)
        logger.info("saving file: attention_scores to atten.npy (%s)", attention_scores.shape)
        np.save(os.path.join(args.predict_dir, "atten.npy"), attention_scores)
    return

def visualize(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)
    

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
            
            
        evaluate = False if args.visualize_train else True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
            
            
        
        #all_attention_scores = np.zeros([len(pred_dataset), args.max_seq_length, args.max_seq_length])
        
        #if args.vis_layer==0 and args.vis_head==0:
        #    all_attention_scores = np.zeros([200, 12, 12, args.max_seq_length, args.max_seq_length])
        
        for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            
            num_layer = args.vis_layer
            num_head = args.vis_head
            
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                attention = outputs[-1][-1]
                #attention.shape = torch.Size([batch_size, num_head, token_length_after, token_length_before)
                #print(len(outputs))
                #print(len(outputs[-1]))
                #print('attention', attention.shape)
                
                attention = torch.stack(outputs[-1], 1)[:, :, -1, 0, :]
                attention = torch.sum(attention, dim=1)
                
                attention = attention.cpu().numpy()[:, (kmer+1):-(kmer+1)]
                attention_scores[index*batch_size:index*batch_size+len(batch[0]),:] = attention
    probs = 0
    return scores, probs
    #return

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if args.do_predict:
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
        required=True,
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
    parser.add_argument("--do_cache", action="store_true", help="whether to only cache train.tsv file")
    parser.add_argument("--do_cache_dev", action="store_true", help="whether to only cache dev.tsv file")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_train_from_scratch", action="store_true", help="Whether to run training from scratch.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction on the given dataset.")
    parser.add_argument("--do_extract", action="store_true", help="Whether to do extraction of embedding vectors on the given dataset.")
    parser.add_argument("--do_extract_train", action="store_true", help="extraction for training and evaluation datasets")
    parser.add_argument("--extract_layer", default=-1, type=int, help="layer of extraction, default(-1) value is the last layer")
    parser.add_argument("--do_visualize", action="store_true", help="Whether to calculate attention score.")
    parser.add_argument("--do_visualize_attentionmap", action="store_true", help="Whether to calculate attention maps.")
    parser.add_argument("--do_analyzeGC", action="store_true", help="Whether to analyze attention based on GC content")
    parser.add_argument("--do_analyzeGC_specific", action="store_true", help="Whether to analyze attention based on GC content at specific head")
    parser.add_argument("--analyze_GCcontent_thresh", default=0.5, type=float, help="threshold percentage of GC content to analyze 0 for calculating mean")
    parser.add_argument("--do_analyze_regiontype", action="store_true", help="Whether to analyze attention based on region types")
    parser.add_argument("--do_analyze_regiontype_specific", action="store_true", help="Whether to analyze attention based on region types at a specific head")
    parser.add_argument("--region_type", default=0, type=int, help="Which region types to analyze 1: 5'UTR, 2: 3'UTR, 3: exon, 4: intron, 5: CDS")
    parser.add_argument("--specific_heads", default="", type=str, help="Which region types to analyze 1: 5'UTR, 2: 3'UTR, 3: exon, 4: intron, 5: CDS")
    parser.add_argument("--do_analyze_regionboundary", action="store_true", help="Whether to analyze regionboundary")
    parser.add_argument("--do_analyze_regionboundary_specific", action="store_true", help="Whether to analyze regionboundary at a specific head")
    parser.add_argument("--do_analyze_rnastructure", action="store_true", help="Whether to analyze rnastructure")
    parser.add_argument("--do_analyze_rnastructure_specific", action="store_true", help="Whether to analyze rnastructure at a specific head")
    parser.add_argument("--correct_position_bias", action="store_true", help="Whether to correct positional effects")
    parser.add_argument("--do_analyze_positional_effect", action="store_true", help="Whether to analyze positional effects")
    parser.add_argument("--positional_effects_dir", default="", type=str, help="directory to save or load positional_effects.npy")
    parser.add_argument("--region_boundaries", default="", type=str, help="directory to save or load positional_effects.npy")
    parser.add_argument("--do_analyze_motif", action="store_true", help="Whether to do motif analysis")
    parser.add_argument("--do_analyze_attentionrollout", action="store_true", help="Whether to do attention flow calculation")
    parser.add_argument("--do_structuralprobe", action="store_true", help="Whether to do structural probing")
    parser.add_argument("--do_attentionprobe", action="store_true", help="Whether to do attention probing")
    parser.add_argument("--probe_train", action="store_true", help="Whether to probe on training dataset")
    parser.add_argument("--probe_matrix_dir", default="", type=str, help="directory where probe matrix is saved")
    parser.add_argument("--probe_epoch", default=1, type=int, help="number of epochs for probe fitting")
    parser.add_argument("--probe_lr", default=0.001, type=float, help="learning rate for structural probing")
    parser.add_argument("--probe_matrix_depth", default=256, type=int, help="depth of probing matrix")
    parser.add_argument("--probe_lambda", default=0.01, type=float, help="L1 regularization for probing tasks")
    parser.add_argument("--path_to_fasta", default=None, type=str, help="path to the folder where fasta file for linear partition will be saved")
    parser.add_argument("--path_to_linearpartition", default='', type=str, help="path to the folder where linearparition is saved")
    parser.add_argument("--vis_layer", default=-1, type=int, help="N th layer to visualize attentions")
    parser.add_argument("--vis_head", default=-1, type=int, help="N th head to visualize attentions")
    parser.add_argument("--output_visimage", action="store_true", help="generate visimage immediately")
    parser.add_argument("--visualize_color", default="", type=str, help="define color schemes for visualization")
    parser.add_argument("--rbp_name", default="", type=str, help="name of rbp to label the output image")
    parser.add_argument("--visualize_train", action="store_true", help="Whether to visualize train.tsv or dev.tsv.")
    parser.add_argument("--do_ensemble_pred", action="store_true", help="Whether to do ensemble prediction with kmer 3456.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
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
    parser.add_argument(
        "--early_stop", default=0, type=int, help="set this to a positive integet if you want to perfrom early stop. The model will stop \
                                                    if the auc keep decreasing early_stop times",
    )
    parser.add_argument(
        "--predict_scan_size",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout rate of attention.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn_dropout", default=0.0, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn", default="lstm", type=str, help="What kind of RNN to use")
    parser.add_argument("--num_rnn_layer", default=2, type=int, help="Number of rnn layers in dnalong model.")
    parser.add_argument("--rnn_hidden", default=768, type=int, help="Number of hidden unit in a rnn layer.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_percent", default=0, type=float, help="Linear warmup over warmup_percent*total_steps.")

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
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--visualize_models", type=int, default=None, help="The model used for visualization. If None, use 3,4,5,6.",
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
    parser.add_argument("--num_gpu", type=int, default=-1, help="number of gpus for multiprocessing")#KEISUKEMP
    parser.add_argument("--num_node", type=int, default=1, help="number of gpus for multiprocessing")#KEISUKEMP
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
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
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
        print('number of gpu: %', torch.cuda.device_count())
        if not (args.no_cuda or args.num_gpu == -1):
            args.n_gpu = args.num_gpu
            args.logging_steps = int(args.logging_steps/args.n_gpu)
    #elif args.local_rank == -1 and args.num_gpu > 1:
    #    device = torch.device("cuda")
    #    args.n_gpu = args.num_gpu
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        import socket
        host = socket.gethostname()
        ip = socket.gethostbyname(host)
        os.environ['MASTER_ADDR'] = str(ip)
        os.environ['MASTER_PORT'] = str(58172)
        torch.distributed.init_process_group(backend="nccl", rank=args.local_rank, world_size=args.num_gpu * args.num_node)
        logger.info('number of gpu: %', torch.cuda.device_count())
        args.n_gpu = args.num_gpu
        args.logging_step = int(args.logging_step/args.n_gpu)
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, num_gpu: %s, num_node: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.num_gpu,
        args.num_node,
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

    if not args.do_visualize and not args.do_ensemble_pred:
        if args.do_extract:
            config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
            output_hidden_states=True
            )
        else:
            config = config_class.from_pretrained(
                args.config_name if args.config_name else args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
        
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        if args.model_type in ["dnalong", "dnalongcat"]:
            assert args.max_seq_length % 512 == 0
        config.split = int(args.max_seq_length/512)
        config.rnn = args.rnn
        config.num_rnn_layer = args.num_rnn_layer
        config.rnn_dropout = args.rnn_dropout
        config.rnn_hidden = args.rnn_hidden

        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        
        if not args.do_train_from_scratch:
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
        else:
            logger.info("Training new model from scratch")
            model = model_class(config=config)
        
        
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)

    
    #only cache examples
    if args.do_cache:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        print('cached:', args.data_dir)
    elif args.do_cache_dev:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
        print('cached:', args.data_dir)
   
    # Training
    if args.do_train and not args.do_cache:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        #if (args.num_gpu=<1 and args.num_node=<1) or args.no_cuda: 
        #    train(args, train_dataset, model, tokenizer)
        #else:
        #    mp.spawn(train, nprocs=args.gpus, args=(args,))

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.task_name != "dna690" and not args.do_cache:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0] and not args.do_cache:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    # Prediction
    predictions = {}
    if args.do_predict and args.local_rank in [-1, 0] and not args.do_cache:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoint = args.output_dir
        logger.info("Predict using the following checkpoint: %s", checkpoint)
        prefix = ''
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        prediction = predict(args, model, tokenizer, prefix=prefix)            

    # Extraction
    extractions = {}
    if args.do_extract and args.local_rank in [-1, 0] and not args.do_cache:
        kmer = 3
        output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
        tokenizer = tokenizer_class.from_pretrained(
            "dna"+str(kmer),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = output_dir
        logger.info("Extract using the following checkpoint: %s", checkpoint)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        model = model_class.from_pretrained(
            checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        config.output_hidden_states = True
        #config.output_attentions = True
        model.to(args.device)
        if args.do_extract_train:
            extract(args, model, tokenizer, prefix=prefix, evaluate=True)
            extract(args, model, tokenizer, prefix=prefix, evaluate=False)
        else:
            extract(args, model, tokenizer, prefix=prefix)

    #Structural probe
    if args.do_structuralprobe and args.local_rank in [-1, 0] and not args.do_cache:
        kmer = 3
        output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
        #checkpoint_name = os.listdir(output_dir)[0]
        #output_dir = os.path.join(output_dir, checkpoint_name)

        tokenizer = tokenizer_class.from_pretrained(
            "dna"+str(kmer),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = output_dir
        logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        config = config_class.from_pretrained(
            output_dir,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        config.output_hidden_states = True
        #config.output_attentions = True
        model = model_class.from_pretrained(
            checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        if args.probe_train:
            probe_matrix = probe_structure(args, model, tokenizer, prefix=prefix, probe_matrix_depth=args.probe_matrix_depth, kmer=kmer, probe_lr=args.probe_lr, probe_lambda=args.probe_lambda, output_example=1)
            probe_matrix = probe_matrix.to('cpu').detach().numpy().copy()
            np.save(os.path.join(args.predict_dir, 'probe_matrix_layer{}.npy'.format(num_layer)), probe_matrix)
        else:
            loss_probe = probe_structure_eval(args, model, tokenizer, prefix=prefix, probe_matrix_depth=args.probe_matrix_depth, kmer=kmer, probe_lr=args.probe_lr, probe_lambda=args.probe_lambda, output_example=1)
        logger.info("finished structural probing:")
        
    # Attention probe
    if args.do_attentionprobe and args.local_rank in [-1, 0] and not args.do_cache:
        kmer = 3
        output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
        #checkpoint_name = os.listdir(output_dir)[0]
        #output_dir = os.path.join(output_dir, checkpoint_name)

        tokenizer = tokenizer_class.from_pretrained(
            "dna"+str(kmer),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = output_dir
        logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        config = config_class.from_pretrained(
            output_dir,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        #config.output_hidden_states = True
        config.output_attentions = True
        model = model_class.from_pretrained(
            checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        if args.probe_train:
            probe_matrix, probe_beta = probe_attention(args, model, tokenizer, prefix=prefix, probe_matrix_depth=args.probe_matrix_depth, kmer=kmer, probe_lr=args.probe_lr, probe_lambda=args.probe_lambda, output_example=10)
            np.save(os.path.join(args.predict_dir, 'probe_attention_layer{}_head{}.npy'.format(args.vis_layer, args.vis_head)),\
                    probe_matrix.cpu().detach().numpy())
            np.save(os.path.join(args.predict_dir, 'probe_beta_layer{}_head{}.npy'.format(args.vis_layer, args.vis_head)),\
                    probe_beta.cpu().detach().numpy())
        else:
            loss_probe = probe_attention_eval(args, model, tokenizer, prefix=prefix, probe_matrix_depth=args.probe_matrix_depth, kmer=kmer, probe_lr=args.probe_lr, probe_lambda=args.probe_lambda, output_example=10)
        
        logger.info("finished attention probing:")
        
    # GCcontent analysis
    if args.do_analyzeGC and args.local_rank in [-1, 0] and not args.do_cache:
        kmer = 3
        output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
        #checkpoint_name = os.listdir(output_dir)[0]
        #output_dir = os.path.join(output_dir, checkpoint_name)

        tokenizer = tokenizer_class.from_pretrained(
            "dna"+str(kmer),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = output_dir
        logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        config = config_class.from_pretrained(
            output_dir,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        #config.output_hidden_states = True
        config.output_attentions = True
        model = model_class.from_pretrained(
            checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        if args.do_analyzeGC_specific:
            gc_vs_attention = analyze_gccontent_specific(args, model, tokenizer, prefix=prefix, kmer=kmer)
            np.save(os.path.join(args.predict_dir, 'analyze_GC_layer{}_head{}.npy'.format(args.vis_layer, args.vis_head)), gc_vs_attention)
        else:
            if args.analyze_GCcontent_thresh == 0:
                gc_list = get_gccontent(args, evaluate=True)
                args.analyze_GCcontent_thresh = np.mean(gc_list)
                print('args.analyze_GCcontent_thresh: ', args.analyze_GCcontent_thresh)
            
            gc_low_matrix, gc_low_count, gc_high_matrix, gc_high_count = analyze_gccontent(args, model, tokenizer, prefix=prefix, kmer=kmer)
            np.save(os.path.join(args.predict_dir, 'analyze_gc_low_thresh{}.npy'.format(int(args.analyze_GCcontent_thresh*100))), gc_low_matrix)
            np.save(os.path.join(args.predict_dir, 'analyze_gc_low_count_thresh{}.npy'.format(int(args.analyze_GCcontent_thresh*100))), gc_low_count)
            np.save(os.path.join(args.predict_dir, 'analyze_gc_high_thresh{}.npy'.format(int(args.analyze_GCcontent_thresh*100))), gc_high_matrix)
            np.save(os.path.join(args.predict_dir, 'analyze_gc_high_count_thresh{}.npy'.format(int(args.analyze_GCcontent_thresh*100))), gc_high_count)
        #np.save(os.path.join(args.predict_dir, 'analyze_GC{}.npy'.format(int(args.analyze_GCcontent_thresh*100))), gc_diff_matrix)
        
        logger.info("finished GC content analysis:")
    
    # regiontype analysis
    if (args.do_analyze_regiontype or args.do_analyze_regiontype_specific) and args.local_rank in [-1, 0] and not args.do_cache:
        kmer = 3
        output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
        #checkpoint_name = os.listdir(output_dir)[0]
        #output_dir = os.path.join(output_dir, checkpoint_name)

        tokenizer = tokenizer_class.from_pretrained(
            "dna"+str(kmer),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = output_dir
        logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        config = config_class.from_pretrained(
            output_dir,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        #config.output_hidden_states = True
        config.output_attentions = True
        model = model_class.from_pretrained(
            checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        if args.do_analyze_regiontype_specific:
            analyze_regiontypes_specific(args, model, tokenizer, prefix=prefix, kmer=kmer)
            '''
            if args.output_visimage:
                num_bins = 15
                attention_max = np.max(regiontype_vs_attention[:,0,1])
                attention_min = 0 #np.min(regiontype_vs_attention[:,0,1])
                attention_thresh = np.linspace(attention_min, attention_max, num_bins+1)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1,1,figsize=(8,6))
                regiontype_values = np.zeros([num_bins])
                count = np.zeros([num_bins])
                for data_point in regiontype_vs_attention:
                    #print(data_point)
                    for i in range(1, len(attention_thresh)):
                        if data_point[0,1] > attention_thresh[i-1] and data_point[0,1] <=attention_thresh[i]:
                            regiontype_values[i-1] += data_point[0,0]
                            count[i-1] += 1
                #print(count)
                #print(np.sum(count))
                regiontype_values = regiontype_values / count / 101
                ax.bar(attention_thresh[:-1], regiontype_values, width=(attention_max-attention_min)*0.05)
                fig.savefig(os.path.join(args.predict_dir, "analyze_regiontype_type{}_layer{}_head{}.png".format(args.region_type, args.vis_layer, args.vis_head)))
            else:
                np.save(os.path.join(args.predict_dir, 'analyze_regiontype_type{}_layer{}_head{}.npy'.format(args.region_type, args.vis_layer, args.vis_head)), regiontype_vs_attention)
            '''
        else:
            regiontype_matrix, regiontype_matrix_negative, regiontype_count, regiontype_count_negative = analyze_regiontypes(args, model, tokenizer, kmer, prefix="")
            heads = ["{}".format(args.rbp_name)]
            regiontypes = ("5'UTR", "3'UTR", "exon", "intron", "CDS")
            for i in range(len(regiontypes)):
                regiontype_matrix[i] = regiontype_matrix[i] / regiontype_count[i]
                regiontype_matrix_negative[i] = regiontype_matrix_negative[i] / regiontype_count_negative[i]
                regiontype_matrix[i] = regiontype_matrix[i] / regiontype_matrix_negative[i]
                max_value, min_value = np.max(regiontype_matrix[i]), np.min(regiontype_matrix[i])
                max_head = np.array(np.where(regiontype_matrix[i]==max_value)) + 1
                min_head = np.array(np.where(regiontype_matrix[i]==min_value)) + 1
                heads.append("region_type: {}, max_head: {}-{}".format(regiontypes[i], max_head[0][0], max_head[1][0]))
                heads.append("region_type: {}, min_head: {}-{}".format(regiontypes[i], min_head[0][0], min_head[1][0]))
            text_path = os.path.join(args.predict_dir, "analyze_regiontype.txt")
            with open(text_path, 'w') as f:
                f.write("\n".join(heads))
                
            if args.output_visimage:
                ax_i = 1
                ax_j = len(regiontypes)
                import matplotlib.pyplot as plt
                import seaborn as sns
                fig, ax = plt.subplots(ax_i, ax_j, figsize=(ax_j*7.5, ax_i*6))
                heatmap_xlabels = list(map(int, np.linspace(1,12,12)))
                heatmap_ylabels = heatmap_xlabels[::-1]
                
                for i in range(len(regiontypes)):
                    sns.heatmap(np.flip(regiontype_matrix[i], axis=0), ax =ax[i], center=1.0)
                    ax[i].set_xlabel("head")
                    ax[i].set_xlabel("layer")
                    ax[i].set_xticklabels(heatmap_xlabels)
                    ax[i].set_yticklabels(heatmap_ylabels)
                    ax[i].set_title("{}: {}".format(args.rbp_name, regiontypes[i]))
                fig.savefig(os.path.join(args.predict_dir, "analyze_regiontype_heatmap.png"))
            else:
                np.save(os.path.join(args.predict_dir, 'analyze_regiontype_type{}.npy'.format(args.region_type)), regiontype_matrix)
                # np.save(os.path.join(args.predict_dir, 'analyze_regiontype_type{}.npy'.format(args.region_type)), regiontype_matrix)
                # np.save(os.path.join(args.predict_dir, 'analyze_regiontype_negative_type{}.npy'.format(args.region_type)), regiontype_matrix_negative)
                np.save(os.path.join(args.predict_dir, 'analyze_regiontype_count_type{}.npy'.format(args.region_type)), regiontype_count)
                np.save(os.path.join(args.predict_dir, 'analyze_regiontype_count_negative_type{}.npy'.format(args.region_type)), regiontype_count_negative)

        logger.info("finished region type analysis:")
    
    # regionboundary analysis
    if (args.do_analyze_regionboundary or args.do_analyze_regionboundary_specific) and args.local_rank in [-1, 0] and not args.do_cache:
        kmer = 3
        output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
        #checkpoint_name = os.listdir(output_dir)[0]
        #output_dir = os.path.join(output_dir, checkpoint_name)

        tokenizer = tokenizer_class.from_pretrained(
            "dna"+str(kmer),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = output_dir
        logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        config = config_class.from_pretrained(
            output_dir,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        #config.output_hidden_states = True
        config.output_attentions = True
        model = model_class.from_pretrained(
            checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        if args.do_analyze_regionboundary_specific:
            analyze_regionboundary_specific(args, model, tokenizer, kmer, prefix="")
            #pass
            '''
            regiontype_vs_attention = analyze_regiontypes_specific(args, model, tokenizer, prefix=prefix, kmer=kmer)
            if args.output_visimage:
                num_bins = 15
                attention_max = np.max(regiontype_vs_attention[:,0,1])
                attention_min = 0 #np.min(regiontype_vs_attention[:,0,1])
                attention_thresh = np.linspace(attention_min, attention_max, num_bins+1)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1,1,figsize=(8,6))
                regiontype_values = np.zeros([num_bins])
                count = np.zeros([num_bins])
                for data_point in regiontype_vs_attention:
                    #print(data_point)
                    for i in range(1, len(attention_thresh)):
                        if data_point[0,1] > attention_thresh[i-1] and data_point[0,1] <=attention_thresh[i]:
                            regiontype_values[i-1] += data_point[0,0]
                            count[i-1] += 1
                #print(count)
                #print(np.sum(count))
                regiontype_values = regiontype_values / count / 101
                ax.bar(attention_thresh[:-1], regiontype_values, width=(attention_max-attention_min)*0.05)
                fig.savefig(os.path.join(args.predict_dir, "analyze_regiontype_type{}_layer{}_head{}.png".format(args.region_type, args.vis_layer, args.vis_head)))
            else:
                np.save(os.path.join(args.predict_dir, 'analyze_regiontype_type{}_layer{}_head{}.npy'.format(args.region_type, args.vis_layer, args.vis_head)), regiontype_vs_attention)
            '''
        else:
            regionboundary_matrix, regionboundary_matrix_negative, regionboundary_count, regionboundary_count_negative, regionboundaries = \
                                                                                                            analyze_regionboundary(args, model, tokenizer, kmer, prefix="")
            regiontypes = ("5'UTR", "3'UTR", "exon", "intron", "CDS", "outside")
            heads = ["{}".format(args.rbp_name)]
            heads.append("regionboundaries: {}".format(regionboundaries))
            regionboundary_matrix2 = regionboundary_matrix.copy()
            for i in range(len(regionboundaries)):
                regionboundary_matrix2[i] = regionboundary_matrix[i] / (regionboundary_matrix[i] + regionboundary_matrix_negative[i])
                if regionboundary_count[i] > 0:
                    regionboundary_matrix[i] = regionboundary_matrix[i] / regionboundary_count[i]
                regionboundary_matrix_negative[i] = regionboundary_matrix_negative[i] / regionboundary_count_negative[i]
                regionboundary_matrix[i] = regionboundary_matrix[i] / regionboundary_matrix_negative[i]
                max_value, min_value = np.max(regionboundary_matrix[i]), np.min(regionboundary_matrix[i])
                max_head = np.array(np.where(regionboundary_matrix[i]==max_value)) + 1
                min_head = np.array(np.where(regionboundary_matrix[i]==min_value)) + 1
                heads.append("region_type: {} and {}, max_head: {}-{}".format(regiontypes[regionboundaries[i][0]], regiontypes[regionboundaries[i][1]], max_head[0][0], max_head[1][0]))
                heads.append("region_type: {} and {}, min_head: {}-{}".format(regiontypes[regionboundaries[i][0]], regiontypes[regionboundaries[i][1]], min_head[0][0], min_head[1][0]))
                text_path = os.path.join(args.predict_dir, "analyze_regionboundary.txt")
                with open(text_path, 'w') as f:
                    f.write("\n".join(heads))
                
            if args.output_visimage:
                import matplotlib.pyplot as plt
                import seaborn as sns
                fig, ax = plt.subplots(1,1, figsize=(7.5, 6))
                heatmap_xlabels = list(map(int, np.linspace(1,12,12)))
                heatmap_ylabels = heatmap_xlabels[::-1]
                
                for i in range(len(regionboundaries)):
                    sns.heatmap(np.flip(regionboundary_matrix, axis=0), ax =ax)
                    ax.set_xlabel("head")
                    ax.set_xlabel("layer")
                    ax.set_xticklabels(heatmap_xlabels)
                    ax.set_yticklabels(heatmap_ylabels)
                    ax.set_title("{}: {} and {}".format(args.rbp_name, regiontypes[regionboundaries[i][0]], regiontypes[regionboundaries[i][1]]))
                    fig.savefig(os.path.join(args.predict_dir, "analyze_regionboundary{}and{}_heatmap.png".format(regiontypes[regionboundaries[i][0]], regiontypes[regionboundaries[i][1]])))
            else:
                np.save(os.path.join(args.predict_dir, 'analyze_regionboundary_all.npy'), regionboundary_matrix)
                np.save(os.path.join(args.predict_dir, 'analyze_regionboundary_all2.npy'), regionboundary_matrix2)
                # np.save(os.path.join(args.predict_dir, 'analyze_regiontype_type{}.npy'.format(args.region_type)), regiontype_matrix)
                # np.save(os.path.join(args.predict_dir, 'analyze_regiontype_negative_type{}.npy'.format(args.region_type)), regiontype_matrix_negative)
                np.save(os.path.join(args.predict_dir, 'analyze_regionboundary_count_all.npy'), regionboundary_count)
                np.save(os.path.join(args.predict_dir, 'analyze_regionboundary_count_negative_all.npy'), regionboundary_count_negative)
                
        logger.info("finished region boundary analysis:")
    
    # rnastructure analysis
    if (args.do_analyze_rnastructure or args.do_analyze_rnastructure_specific) and args.local_rank in [-1, 0] and not args.do_cache:
        kmer = 3
        output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
        #checkpoint_name = os.listdir(output_dir)[0]
        #output_dir = os.path.join(output_dir, checkpoint_name)

        tokenizer = tokenizer_class.from_pretrained(
            "dna"+str(kmer),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = output_dir
        logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        config = config_class.from_pretrained(
            output_dir,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        #config.output_hidden_states = True
        config.output_attentions = True
        model = model_class.from_pretrained(
            checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        if args.do_analyze_rnastructure_specific:
            analyze_rnastructure_specific(args, model, tokenizer, kmer, prefix="")
            '''
            regiontype_vs_attention = analyze_regiontypes_specific(args, model, tokenizer, prefix=prefix, kmer=kmer)
            if args.output_visimage:
                num_bins = 15
                attention_max = np.max(regiontype_vs_attention[:,0,1])
                attention_min = 0 #np.min(regiontype_vs_attention[:,0,1])
                attention_thresh = np.linspace(attention_min, attention_max, num_bins+1)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1,1,figsize=(8,6))
                regiontype_values = np.zeros([num_bins])
                count = np.zeros([num_bins])
                for data_point in regiontype_vs_attention:
                    #print(data_point)
                    for i in range(1, len(attention_thresh)):
                        if data_point[0,1] > attention_thresh[i-1] and data_point[0,1] <=attention_thresh[i]:
                            regiontype_values[i-1] += data_point[0,0]
                            count[i-1] += 1
                #print(count)
                #print(np.sum(count))
                regiontype_values = regiontype_values / count / 101
                ax.bar(attention_thresh[:-1], regiontype_values, width=(attention_max-attention_min)*0.05)
                fig.savefig(os.path.join(args.predict_dir, "analyze_regiontype_type{}_layer{}_head{}.png".format(args.region_type, args.vis_layer, args.vis_head)))
            else:
                np.save(os.path.join(args.predict_dir, 'analyze_regiontype_type{}_layer{}_head{}.npy'.format(args.region_type, args.vis_layer, args.vis_head)), regiontype_vs_attention)
            '''
        else:
            rnastructure_matrix, rnastructure_matrix_negative, rnastructure_count, rnastructure_count_negative = \
                                                                                                            analyze_rnastructure(args, model, tokenizer, kmer, prefix="")
            rnastructure_matrix2 = rnastructure_matrix.copy()
            heads = ["{}".format(args.rbp_name)]
            structuretypes = ("F(dangling start)", "T(dangling end)", "I(internal loop)", "H(hairpin loop)", "M(multi loop)", "S(stem)")
            for i in range(len(structuretypes)):
                rnastructure_matrix2[i] = rnastructure_matrix[i] / (rnastructure_matrix[i] + rnastructure_matrix_negative[i])
                rnastructure_matrix[i] = rnastructure_matrix[i] / rnastructure_count[i]
                rnastructure_matrix_negative[i] = rnastructure_matrix_negative[i] / rnastructure_count_negative[i]
                rnastructure_matrix[i] = rnastructure_matrix[i] / rnastructure_matrix_negative[i]
                max_value, min_value = np.max(rnastructure_matrix[i]), np.min(rnastructure_matrix[i])
                max_head = np.array(np.where(rnastructure_matrix[i]==max_value)) + 1
                min_head = np.array(np.where(rnastructure_matrix[i]==min_value)) + 1
                heads.append("rnastructure_type: {}, max_head: {}-{}".format(structuretypes[i], max_head[0][0], max_head[1][0]))
                heads.append("rnastructure_type: {}, min_head: {}-{}".format(structuretypes[i], min_head[0][0], min_head[1][0]))
            text_path = os.path.join(args.predict_dir, "analyze_rnastructure.txt")
            with open(text_path, 'w') as f:
                f.write("\n".join(heads))
                
            if args.output_visimage:
                ax_i = 1
                ax_j = len(structuretypes)
                import matplotlib.pyplot as plt
                import seaborn as sns
                fig, ax = plt.subplots(ax_i, ax_j, figsize=(ax_j*7.5, ax_i*6))
                heatmap_xlabels = list(map(int, np.linspace(1,12,12)))
                heatmap_ylabels = heatmap_xlabels[::-1]
                
                for i in range(len(structuretypes)):
                    sns.heatmap(np.flip(rnastructure_matrix[i], axis=0), ax =ax[i])
                    ax[i].set_xlabel("head")
                    ax[i].set_xlabel("layer")
                    ax[i].set_xticklabels(heatmap_xlabels)
                    ax[i].set_yticklabels(heatmap_ylabels)
                    ax[i].set_title("{}: {}".format(args.rbp_name, structuretypes[i]))
                fig.savefig(os.path.join(args.predict_dir, "analyze_rnastructure_heatmap.png"))
            else:
                np.save(os.path.join(args.predict_dir, 'analyze_rnastructure.npy'), rnastructure_matrix)
                np.save(os.path.join(args.predict_dir, 'analyze_rnastructure2.npy'), rnastructure_matrix2)
                # np.save(os.path.join(args.predict_dir, 'analyze_regiontype_type{}.npy'.format(args.region_type)), regiontype_matrix)
                # np.save(os.path.join(args.predict_dir, 'analyze_regiontype_negative_type{}.npy'.format(args.region_type)), regiontype_matrix_negative)
                np.save(os.path.join(args.predict_dir, 'analyze_rnastructure_count_type{}.npy'.format(args.region_type)), rnastructure_count)
                np.save(os.path.join(args.predict_dir, 'analyze_rnastructure_count_negative_type{}.npy'.format(args.region_type)), rnastructure_count_negative)

        logger.info("finished rna structure analysis:")
    
    # positional effect
    if args.do_analyze_positional_effect and args.local_rank in [-1, 0] and not args.do_cache:
        kmer = 3
        output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
        #checkpoint_name = os.listdir(output_dir)[0]
        #output_dir = os.path.join(output_dir, checkpoint_name)

        tokenizer = tokenizer_class.from_pretrained(
            "dna"+str(kmer),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = output_dir
        logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        config = config_class.from_pretrained(
            output_dir,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        #config.output_hidden_states = True
        config.output_attentions = True
        model = model_class.from_pretrained(
            checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        positional_effects = analyze_positional_effects(args, model, tokenizer, kmer, prefix="")
        np.save(os.path.join(args.positional_effects_dir, 'positional_effects.npy'), positional_effects)
    
    # motif analysis
    if args.do_analyze_motif and args.local_rank in [-1, 0] and not args.do_cache:
        kmer = 3
        output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
        #checkpoint_name = os.listdir(output_dir)[0]
        #output_dir = os.path.join(output_dir, checkpoint_name)

        tokenizer = tokenizer_class.from_pretrained(
            "dna"+str(kmer),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = output_dir
        logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        config = config_class.from_pretrained(
            output_dir,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        #config.output_hidden_states = True
        config.output_attentions = True
        model = model_class.from_pretrained(
            checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        analyze_motif(args, model, tokenizer, prefix=prefix, kmer=kmer)
        logger.info("finished motif analysis")
    
    # attention flow analysis
    if args.do_analyze_attentionrollout and args.local_rank in [-1, 0] and not args.do_cache:
        kmer = 3
        output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
        #checkpoint_name = os.listdir(output_dir)[0]
        #output_dir = os.path.join(output_dir, checkpoint_name)

        tokenizer = tokenizer_class.from_pretrained(
            "dna"+str(kmer),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = output_dir
        logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        config = config_class.from_pretrained(
            output_dir,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        #config.output_hidden_states = True
        config.output_attentions = True
        model = model_class.from_pretrained(
            checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        analyze_attentionrollout(args, model, tokenizer, prefix=prefix, kmer=kmer)
        logger.info("finished attention rollout analysis")
        
        
    # Visualize
    if (args.do_visualize or args.do_visualize_attentionmap) and args.local_rank in [-1, 0] and not args.do_cache:
        visualization_models = [3,4,5,6] if not args.visualize_models else [args.visualize_models]

        scores = None
        all_probs = None

        for kmer in visualization_models:
            output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
            #checkpoint_name = os.listdir(output_dir)[0]
            #output_dir = os.path.join(output_dir, checkpoint_name)
            
            tokenizer = tokenizer_class.from_pretrained(
                "dna"+str(kmer),
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            checkpoint = output_dir
            logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            config = config_class.from_pretrained(
                output_dir,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            config.output_attentions = True
            model = model_class.from_pretrained(
                checkpoint,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.to(args.device)
            attention_scores, probs = visualize(args, model, tokenizer, prefix=prefix, kmer=kmer)
            logger.info("finished caluculating attention scores:", attention_scores.shape)
            
            if scores is not None:
                #all_probs += probs
                scores += attention_scores
            else:
                #all_probs = deepcopy(probs)
                scores = deepcopy(attention_scores)
        
        if args.do_visualize:
            logger.info("saving file: scores to atten.npy (%s)", scores.shape)
            #logger.info("saving file: all_probs to pred_results.npy (%s)", all_probs.shape)
            #all_probs = all_probs/float(len(visualization_models))
            np.save(os.path.join(args.predict_dir, "atten.npy"), scores)
            #np.save(os.path.join(args.predict_dir, "pred_results.npy"), all_probs)

    # ensemble prediction
    if args.do_ensemble_pred and args.local_rank in [-1, 0] and not args.do_cache:

        for kmer in range(3,7):
            output_dir = os.path.join(args.output_dir, str(kmer))
            tokenizer = tokenizer_class.from_pretrained(
                "dna"+str(kmer),
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            checkpoint = output_dir
            logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            config = config_class.from_pretrained(
                config_path,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            config.output_attentions = True
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.to(args.device)
            if kmer == 3:
                args.data_dir = os.path.join(args.data_dir, str(kmer))
            else:
                args.data_dir = args.data_dir.replace("/"+str(kmer-1), "/"+str(kmer))

            if args.result_dir.split('/')[-1] == "test.npy":
                results, eval_task, _, out_label_ids, probs = evaluate(args, model, tokenizer, prefix=prefix)
            elif args.result_dir.split('/')[-1] == "train.npy":
                results, eval_task, _, out_label_ids, probs = evaluate(args, model, tokenizer, prefix=prefix, evaluate=False)
            else:
                raise ValueError("file name in result_dir should be either test.npy or train.npy")

            if kmer == 3:
                all_probs = deepcopy(probs)
                cat_probs = deepcopy(probs)
            else:
                all_probs += probs
                cat_probs = np.concatenate((cat_probs, probs), axis=1)
            print(cat_probs[0])
        

        all_probs = all_probs / 4.0
        all_preds = np.argmax(all_probs, axis=1)
        
        # save label and data for stuck ensemble
        labels = np.array(out_label_ids)
        labels = labels.reshape(labels.shape[0],1)
        data = np.concatenate((cat_probs, labels), axis=1)
        random.shuffle(data)
        root_path = args.result_dir.replace(args.result_dir.split('/')[-1],'')
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        # data_path = os.path.join(root_path, "data")
        # pred_path = os.path.join(root_path, "pred")
        # if not os.path.exists(data_path):
        #     os.makedirs(data_path)
        # if not os.path.exists(pred_path):
        #     os.makedirs(pred_path)
        # np.save(os.path.join(data_path, args.result_dir.split('/')[-1]), data)
        # np.save(os.path.join(pred_path, "pred_results.npy", all_probs[:,1]))
        np.save(args.result_dir, data)
        ensemble_results = compute_metrics(eval_task, all_preds, out_label_ids, all_probs[:,1])
        logger.info("***** Ensemble results {} *****".format(prefix))
        for key in sorted(ensemble_results.keys()):
            logger.info("  %s = %s", key, str(ensemble_results[key]))    


            


    return results


if __name__ == "__main__":
    main()
