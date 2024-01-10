import os
from tucore_gcn_bert_tokenizer import SpeakerBertTokenizer
from tucore_gcn_bert_pipeline import create_model_inputs
from tucore_gcn_bert_modelling import (
    TUCOREGCN_BertConfig,
    TUCOREGCN_BertForSequenceClassification,
)
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
from tqdm import trange
import json
from tucore_gcn_bert_processor import (
    SpeakerRelation,
    Conversation,
    DialogRE,
)
import random
from datasets.utils.info_utils import VerificationMode
import datasets
import csv
import os
import logging
import argparse
import random
import pickle

import numpy as np
import torch

import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_

from TUCOREGCN_BERT import TUCOREGCN_BERT as other_tucore
def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}


class BERTAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(BERTAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


n_classes = {
    "DialogRE": 36,
    "MELD": 7,
    "EmoryNLP": 7,
    "DailyDialog": 7,
}

reverse_order = False
sa_step = False


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def f1_eval(logits, labels_all):
    def getpred(result, T1=0.5, T2=0.4):
        ret = []
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [36]
                else:
                    r += [maxj]
            ret += [r]
        return ret

    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0

        for i in range(len(data)):
            for id in data[i]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[i]:
                        correct_sys += 1

            for id in devp[i]:
                if id != 36:
                    all_sys += 1

        precision = 1 if all_sys == 0 else correct_sys / all_sys
        recall = 0 if correct_gt == 0 else correct_sys / correct_gt
        f_1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall != 0
            else 0
        )
        return f_1

    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))

    labels = []
    for la in labels_all:
        label = []
        for i in range(36):
            if la[i] == 1:
                label += [i]
        if len(label) == 0:
            label = [36]
        labels += [label]
    assert len(labels) == len(logits)

    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2 / 100.0)
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2 / 100.0

    return bestf_1, bestT2


def accuracy(out, labels):
    out = out.reshape(-1)
    out = 1 / (1 + np.exp(-out))
    return np.sum((out > 0.5) == (labels > 0.5)) / 36


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """Utility function for optimize_on_cpu and 16-bits training.
    Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(
        named_params_optimizer, named_params_model
    ):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(
    named_params_optimizer, named_params_model, test_nan=False
):
    """Utility function for optimize_on_cpu and 16-bits training.
    Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(
        named_params_optimizer, named_params_model
    ):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(
                param_opti.data.new().resize_(*param_opti.data.size())
            )
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan

def get_logits4eval(model, dataset, batch_size, savefile, device):
    model.eval()
    logits_all = []
    n_batches = len(dataset) // batch_size
    for step in tqdm(range(n_batches), desc="Iteration"):
        shard = dataset.shard(num_shards=n_batches, index=step)
        label_ids = torch.LongTensor(shard["label_ids"]).contiguous().to(device).float()
        input_ids = torch.LongTensor(shard["input_ids"]).contiguous().to(device)
        segment_ids = torch.LongTensor(shard["segment_ids"]).contiguous().to(device)
        input_masks = torch.LongTensor(shard["input_mask"]).contiguous().to(device)
        mention_ids = torch.LongTensor(shard["mention_ids"]).contiguous().to(device)
        speaker_ids = torch.LongTensor(shard["speaker_ids"]).contiguous().to(device)
        turn_mask = torch.LongTensor(shard["turn_masks"]).contiguous().to(device)
        # forgot to flatten the list 1
        graphs = [pickle.loads(g)[0].to(device) for g in shard["graph"]]

        with torch.no_grad():
            output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, speaker_ids=speaker_ids, graphs=graphs, mention_ids=mention_ids, labels=label_ids, turn_mask=turn_mask)
            logits = output.logits

        logits = logits.detach().cpu().numpy()
        for i in range(len(logits)):
            logits_all += [logits[i]]

    with open(savefile, "w") as f:
        for i in range(len(logits_all)):
            for j in range(len(logits_all[i])):
                f.write(str(logits_all[i][j]))
                if j == len(logits_all[i])-1:
                    f.write("\n")
                else:
                    f.write(" ")

"""
python tucore_gcn_transformers/tucore_gcn_bert_train.py --do_train --do_eval --output_dir tucore_gcn_bert_test1
python tucore_gcn_transformers/tucore_gcn_bert_train.py --do_train --do_eval --output_dir tucore_gcn_bert_test1 --num_train_epochs 1
python tucore_gcn_transformers/tucore_gcn_bert_train.py --do_train --do_eval --output_dir tucore_gcn_bert_test1 --num_train_epochs 1 --resume [X]
python tucore_gcn_transformers/tucore_gcn_bert_train.py --do_train --do_eval --do_test_eval 1 --output_dir TUCOREGCN_BERT_DialogRE --num_train_epochs 1 --resume 69 --eval_batch_size 32
4.0gb
2.9gb
seq_length of 512
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="datasets/DialogRE",
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--config_file",
        default="tucore_gcn_transformers/tucore_gcn_bert_mlc.json",
        type=str,
        help="The config json file corresponding to the pre-trained model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--data_name",
        default="DialogRE",
        type=str,
        help="The name of the dataset to train.",
    )
    parser.add_argument(
        "--encoder_type",
        default="BERT",
        type=str,
        help="The type of pre-trained model.",
    )
    parser.add_argument(
        "--vocab_file",
        default="pre-trained_model/BERT/vocab.txt",
        type=str,
        help="The vocabulary file that the model was trained on.",
    )
    parser.add_argument(
        "--merges_file",
        default=None,
        type=str,
        help="The merges file that the RoBERTa model was trained on.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--init_checkpoint",
        default="pre-trained_model/BERT/pytorch_model.bin",
        type=str,
        help="Initial checkpoint (usually from a pre-trained model).",
    )
    parser.add_argument(
        "--do_lower_case",
        default=False,
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--do_train",
        default=False,
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--do_eval",
        default=False,
        action="store_true",
        help="Whether to run eval on the dev set.",
    )
    parser.add_argument(
        "--do_test_eval",
        default=0,
        type=int,
        help="Whether to run eval on the dev set.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=12,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=12, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--save_checkpoints_steps",
        default=1000,
        type=int,
        help="How often to save the model checkpoint.",
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        action="store_true",
        help="Whether not to use CUDA when available",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=666, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--optimize_on_cpu",
        default=False,
        action="store_true",
        help="Whether to perform optimization and keep the optimizer averages on CPU",
    )
    parser.add_argument(
        "--fp16",
        default=False,
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=128,
        help="Loss scaling, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument(
        "--resume",
        default=-1,
        type=int,
        help="Whether to resume the training.",
    )
    parser.add_argument(
        "--f1eval",
        default=True,
        action="store_true",
        help="Whether to use f1 for dev evaluation during training.",
    )

    args = parser.parse_args()
    # DialogRE, EmoryNLP, DailyDialog, MELD
    if args.data_name not in n_classes:
        raise ValueError("Data not found: %s" % (args.data_name))
    n_class = n_classes[args.data_name]
    # Too poor to afford more than 1 GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = 1
    logger.info(
        "device %s n_gpu %d distributed training %r",
        device,
        n_gpu,
        bool(args.local_rank != -1),
    )
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    # Consistently get subpar f1c
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # just comment out evaluation code for testing
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    # Need to Implement Roberta TUCORE-GCN config
    config = TUCOREGCN_BertConfig.from_json_file(args.config_file)
    #
    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, config.max_position_embeddings
            )
        )
    # Don't overwrite last session's test model
    if os.path.exists(args.output_dir) and "model.pt" in os.listdir(args.output_dir):
        if args.do_train and args.resume==-1:
            raise ValueError(
                "Output directory ({}) already exists and is not empty.".format(
                    args.output_dir
                )
            )
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    # Custom speaker tokenizer with using deprecated workaround but who cares
    tokenizer = SpeakerBertTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case
    )
    # updated datasets repo
    tucore_data = datasets.load_from_disk("./datasets/DialogRE/arrow/")
    train_set, test_set = tucore_data["train"], tucore_data["test"]
    # below sets train_set and test_set to small subset for testings
    #train_set = train_set.shard(num_shards=100, index=0)
    #test_set = test_set.shard(num_shards=40, index=0)
    num_train_steps = int(
        len(train_set)
        / args.train_batch_size
        / args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    '''
    model = other_tucore(config, 36)
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(
            torch.load(args.init_checkpoint, map_location="cpu"), strict=False
        )
    '''
    #'''
    model = TUCOREGCN_BertForSequenceClassification(config)
    if args.init_checkpoint is not None:
        model.tucoregcn_bert.bert.load_state_dict(
            torch.load(args.init_checkpoint, map_location="cpu"), strict=False
        )
    #'''
    model.to(device)
    no_decay = ["bias", "weight", "bias"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if n not in no_decay],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if n in no_decay],
            "weight_decay_rate": 0.0,
        },
    ]
    optimizer = BERTAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        t_total=num_train_steps,
    )
    epoch_start = 0
    global_step = 0
    if args.resume!=-1:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f"{args.resume}.pt")))
        epoch_start = args.resume+1
    if args.do_train:
        best_metric = 0
        train_n_batches = len(train_set) // args.train_batch_size
        test_n_batches = len(test_set) // args.eval_batch_size
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        for epoch in trange(epoch_start, int(args.num_train_epochs)+epoch_start, desc="Epoch"):
            if args.do_test_eval==0:
                model.train()
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0

                for step in tqdm(range(0, train_n_batches)):
                    shard = train_set.shard(num_shards=train_n_batches, index=step)
                    # need to reload dataset with label_ids bc I forgor
                    label_ids = torch.LongTensor(shard["label_ids"]).contiguous().to(device).float()
                    input_ids = torch.LongTensor(shard["input_ids"]).contiguous().to(device)
                    segment_ids = torch.LongTensor(shard["segment_ids"]).contiguous().to(device)
                    input_masks = torch.LongTensor(shard["input_mask"]).contiguous().to(device)
                    mention_ids = torch.LongTensor(shard["mention_ids"]).contiguous().to(device)
                    speaker_ids = torch.LongTensor(shard["speaker_ids"]).contiguous().to(device)
                    turn_mask = torch.LongTensor(shard["turn_masks"]).contiguous().to(device)
                    # forgot to flatten the list 1
                    graphs = [pickle.loads(g)[0].to(device) for g in shard["graph"]]
                    outputs= model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_masks,
                        speaker_ids=speaker_ids,
                        graphs=graphs,
                        mention_ids=mention_ids,
                        labels=label_ids,
                        turn_mask=turn_mask,
                    )
                    loss = outputs.loss
                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.fp16 and args.loss_scale != 1.0:
                        # rescale loss for fp16 training
                        # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                        loss = loss * args.loss_scale
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16 or args.optimize_on_cpu:
                            if args.fp16 and args.loss_scale != 1.0:
                                # scale down gradients for fp16 training
                                for param in model.parameters():
                                    param.grad.data = param.grad.data / args.loss_scale
                            is_nan = set_optimizer_params_grad(
                                param_optimizer, model.named_parameters(), test_nan=True
                            )
                            if is_nan:
                                logger.info(
                                    "FP16 TRAINING: Nan in gradients, reducing loss scaling"
                                )
                                args.loss_scale = args.loss_scale / 2
                                model.zero_grad()
                                continue
                            optimizer.step()
                            copy_optimizer_params_to_model(
                                model.named_parameters(), param_optimizer
                            )
                        else:
                            optimizer.step()
                        model.zero_grad()
                        global_step += 1
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            logits_all = []
            labels_all = []
            for step in tqdm(range(0, test_n_batches)):
                shard = test_set.shard(num_shards=test_n_batches, index=step)
                # need to reload dataset with label_ids bc I forgor
                label_ids = torch.Tensor(shard["label_ids"]).contiguous().to(device).float()
                input_ids = torch.LongTensor(shard["input_ids"]).contiguous().to(device)
                segment_ids = torch.LongTensor(shard["segment_ids"]).contiguous().to(device)
                input_masks = torch.LongTensor(shard["input_mask"]).contiguous().to(device)
                mention_ids = torch.LongTensor(shard["mention_ids"]).contiguous().to(device)
                speaker_ids = torch.LongTensor(shard["speaker_ids"]).contiguous().to(device)
                turn_mask = torch.LongTensor(shard["turn_masks"]).contiguous().to(device)
                # forgot to flatten the list 2
                graphs = [pickle.loads(g)[0].to(device) for g in shard["graph"]]
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_masks,
                        speaker_ids=speaker_ids,
                        graphs=graphs,
                        mention_ids=mention_ids,
                        labels=label_ids,
                        turn_mask=turn_mask,
                    )
                tmp_eval_loss, logits = outputs.loss, outputs.logits
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to("cpu").numpy()
                for i in range(len(logits)):
                    logits_all += [logits[i]]
                for i in range(len(label_ids)):
                    labels_all.append(label_ids[i])

                tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1
            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            if args.do_train and args.do_test_eval==0:
                result = {
                    "eval_loss": eval_loss,
                    "global_step": global_step,
                    "loss": tr_loss / nb_tr_steps,
                }
            else:
                result = {"eval_loss": eval_loss}

            if args.f1eval:
                eval_f1, eval_T2 = f1_eval(logits_all, labels_all)
                result["f1"] = eval_f1
                result["T2"] = eval_T2

            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            if args.f1eval:
                if eval_f1 >= best_metric:
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.output_dir, "model_best.pt"),
                    )
                    best_metric = eval_f1
            else:
                if eval_accuracy >= best_metric:
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.output_dir, "model_best.pt"),
                    )
                    best_metric = eval_accuracy
            if args.do_test_eval==0:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, f"{epoch}.pt"),
                )
        model.load_state_dict(
            torch.load(os.path.join(args.output_dir, "model_best.pt"))
        )
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))


_CITATION = """\
@inproceedings{yu2020dialogue,
  title={Dialogue-Based Relation Extraction},
  author={Yu, Dian and Sun, Kai and Cardie, Claire and Yu, Dong},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020},
  url={https://arxiv.org/abs/2004.08056v1}
}
"""

_DESCRIPTION = """\
DialogRE is the first human-annotated dialogue based relation extraction (RE) dataset aiming
to support the prediction of relation(s) between two arguments that appear in a dialogue.
The dataset annotates all occurrences of 36 possible relation types that exist between pairs
of arguments in the 1,788 dialogues originating from the complete transcripts of Friends.
"""

_HOMEPAGE = "https://github.com/nlpdata/dialogre"

_LICENSE = "https://github.com/nlpdata/dialogre/blob/master/license.txt"

_URL = "https://raw.githubusercontent.com/nlpdata/dialogre/master/data_v2/en/data/"

_URLs = {
    "train": _URL + "train.json",
    "dev": _URL + "dev.json",
    "test": _URL + "test.json",
}


class TUCOREGCNDialogREDatasetConfig(datasets.BuilderConfig):
    r"""BuilderConfig for DialogRE"""

    def __init__(self, **kwargs):
        r"""BuilderConfig for DialogRE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TUCOREGCNDialogREDatasetConfig, self).__init__(**kwargs)


class TUCOREGCNDialogREDataset(datasets.GeneratorBasedBuilder):
    r"""DialogRE: Human-annotated dialogue-based relation extraction dataset Version 2

    Adapted from https://huggingface.co/datasets/dialog_re, https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py

    Dataset Loader for DialogRE

    Modifications Summary:
        Added preprocessing to _generate_examples.
    """

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        TUCOREGCNDialogREDatasetConfig(
            name="tucore_gcn_dialog_re",
            version=datasets.Version("1.0.0"),
            description="DialogRE dataset formatted for Dialog Relation Extraction task by TUCORE-GCN",
        ),
    ]

    """
    "tokens": tokens,
    "input_ids": input_ids,
    "input_mask": input_mask,
    "segment_ids": segment_ids,
    "speaker_ids": speaker_ids,
    "mention_ids": mention_ids,
    "turn_masks": turn_masks,
    "graph": graph,
    """

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "label_ids": datasets.Sequence(datasets.Value("int32")),
                    "input_ids": datasets.Sequence(datasets.Value("int32")),
                    "input_mask": datasets.Sequence(datasets.Value("int32")),
                    "segment_ids": datasets.Sequence(datasets.Value("int32")),
                    "speaker_ids": datasets.Sequence(datasets.Value("int32")),
                    "mention_ids": datasets.Sequence(datasets.Value("int32")),
                    "turn_masks": datasets.Sequence(
                        datasets.Sequence(datasets.Value("bool"))
                    ),
                    "graph": datasets.Value("binary"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["train"]),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["test"]),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["dev"]),
                    "split": "dev",
                },
            ),
        ]

    def _download_and_prepare(
        self, dl_manager, verification_mode, **prepare_splits_kwargs
    ):
        super()._download_and_prepare(
            dl_manager,
            VerificationMode.NO_CHECKS,
            **prepare_splits_kwargs,
        )

    def _generate_examples(
        self, filepath, split, max_seq_length=512, for_f1c=False, old_behaviour=False, shuffle_train=True
    ):
        r"""Yields examples."""
        speaker_tokenizer = SpeakerBertTokenizer.from_pretrained("bert-base-uncased")
        with open(filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            if split == "train" and not for_f1c and shuffle_train:
                random.shuffle(dataset)
            for tqdm_idx, entry in tqdm_notebook(
                enumerate(dataset), total=len(dataset)
            ):
                dialog_raw = entry[0]
                relation_data = entry[1]
                relations = [
                    SpeakerRelation(relation["x"], relation["y"], relation["rid"])
                    for relation in relation_data
                ]
                c = Conversation(dialog_raw, relations)
                speaker_relations_iterator = enumerate(c.speaker_relations)
                while True:
                    idx, relation = next(speaker_relations_iterator, (-1, -1))
                    if idx == -1:
                        break
                    ret_dialog, ret_relation = c.build_input_with_relation(
                        relation,
                        speaker_tokenizer,
                        max_seq_length,
                        for_f1c,
                        old_behaviour,
                    ).values()
                    if ret_dialog != "":
                        entry = {
                            "dialog": ret_dialog,
                            "relation": ret_relation,
                        }
                        (
                            tokens,
                            label_ids,
                            input_ids,
                            input_mask,
                            segment_ids,
                            speaker_ids,
                            mention_ids,
                            turn_masks,
                            graph,
                        ) = create_model_inputs(
                            speaker_tokenizer.tokenize(entry["dialog"]),
                            speaker_tokenizer.tokenize(entry["relation"]["entity_1"]),
                            speaker_tokenizer.tokenize(entry["relation"]["entity_2"]),
                            speaker_tokenizer,
                            entry,
                            old_behaviour,
                            36,
                            max_seq_length,
                            True,
                        )
                        yield idx, {
                            "tokens": tokens[0],
                            "label_ids": label_ids[0],
                            "input_ids": input_ids[0],
                            "input_mask": input_mask[0],
                            "segment_ids": segment_ids[0],
                            "speaker_ids": speaker_ids[0],
                            "mention_ids": mention_ids[0],
                            "turn_masks": turn_masks[0],
                            "graph": pickle.dumps(graph),
                        }


if __name__ == "__main__":
    main()
