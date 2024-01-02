import os
from optimization import BERTAdam
from tucore_gcn_transformers.tucore_gcn_bert_tokenizer import SpeakerBertTokenizer
from tucore_gcn_transformers.tucore_gcn_bert_pipeline import create_model_inputs
from tucore_gcn_transformers.tucore_gcn_bert_modelling import (
    TUCOREGCN_BertConfig,
    TUCOREGCN_BertForSequenceClassification,
)
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
from tqdm import trange
import json
from tucore_gcn_transformers.tucore_gcn_bert_processor import (
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
    def getpred(result, T1 = 0.5, T2 = 0.4):
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

        precision = 1 if all_sys == 0 else correct_sys/all_sys
        recall = 0 if correct_gt == 0 else correct_sys/correct_gt
        f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
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
    assert(len(labels) == len(logits))
    
    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2/100.)
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2/100.

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--data_name",
        default=None,
        type=str,
        required=True,
        help="The name of the dataset to train.",
    )
    parser.add_argument(
        "--encoder_type",
        default=None,
        type=str,
        required=True,
        help="The type of pre-trained model.",
    )
    parser.add_argument(
        "--vocab_file",
        default=None,
        type=str,
        required=True,
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
        default=None,
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
        default=128,
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
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
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
        default=False,
        action="store_true",
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
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    logger.info(
        "device %s n_gpu %d distributed training %r",
        device,
        n_gpu,
        bool(args.local_rank != -1),
    )
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
        if args.do_train and not args.resume:
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
    tucore_dataset = TUCOREGCNDataset()
    tucore_dataset.download_and_prepare()
    tucore_data = tucore_dataset.as_dataset()
    train_set, test_set = tucore_data["train"], tucore_data["test"]
    num_train_steps = int(
        len(train_set)
        / args.train_batch_size
        / args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    model = TUCOREGCN_BertForSequenceClassification(config, n_class)
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(
            torch.load(args.init_checkpoint, map_location="cpu"), strict=False
        )
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
    global_step = 0
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))
    if args.do_train:
        best_metric = 0
        train_n_batches = len(train_set) // args.train_batch_size
        test_n_batches = len(test_set) // args.train_batch_size
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step in trange(start=0, stop=train_n_batches):
                shard = train_set.shard(num_shards=train_n_batches, index=step)
                # need to reload dataset with label_ids bc I forgor
                label_ids = torch.LongTensor(shard["label_ids"]).to(device)
                input_ids = torch.LongTensor(shard["input_ids"]).to(device)
                segment_ids = torch.LongTensor(shard["segment_ids"]).to(device)
                input_masks = torch.LongTensor(shard["input_mask"]).to(device)
                mention_ids = torch.LongTensor(shard["mention_ids"]).to(device)
                speaker_ids = torch.LongTensor(shard["speaker_ids"]).to(device)
                turn_mask = torch.LongTensor(shard["turn_masks"]).to(device)
                graphs = [pickle.loads(g) for g in shard["graph"]]

                loss, _ = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_masks,
                    speaker_ids=speaker_ids,
                    graphs=graphs,
                    mention_id=mention_ids,
                    labels=label_ids,
                    turn_mask=turn_mask,
                )
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
            for step in trange(start=0, stop=test_n_batches):
                shard = test_set.shard(num_shards=test_n_batches, index=step)
                # need to reload dataset with label_ids bc I forgor
                label_ids = torch.LongTensor(shard["label_ids"]).to(device)
                input_ids = torch.LongTensor(shard["input_ids"]).to(device)
                segment_ids = torch.LongTensor(shard["segment_ids"]).to(device)
                input_masks = torch.LongTensor(shard["input_mask"]).to(device)
                mention_ids = torch.LongTensor(shard["mention_ids"]).to(device)
                speaker_ids = torch.LongTensor(shard["speaker_ids"]).to(device)
                turn_mask = torch.LongTensor(shard["turn_masks"]).to(device)
                graphs = [pickle.loads(g) for g in shard["graph"]]
                with torch.no_grad():
                    tmp_eval_loss, logits = model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_masks,
                        speaker_ids=speaker_ids,
                        graphs=graphs,
                        mention_id=mention_ids,
                        labels=label_ids,
                        turn_mask=turn_mask,
                    )

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

            if args.do_train:
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
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_best.pt")))
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


class TUCOREGCNDatasetConfig(datasets.BuilderConfig):
    r"""BuilderConfig for DialogRE"""

    def __init__(self, **kwargs):
        r"""BuilderConfig for DialogRE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TUCOREGCNDatasetConfig, self).__init__(**kwargs)


class TUCOREGCNDataset(datasets.GeneratorBasedBuilder):
    r"""DialogRE: Human-annotated dialogue-based relation extraction dataset Version 2

    Adapted from https://huggingface.co/datasets/dialog_re, https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py

    Dataset Loader for DialogRE

    Modifications Summary:
        Added preprocessing to _generate_examples.
    """

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        TUCOREGCNDatasetConfig(
            name="dialog_re",
            version=datasets.Version("1.1.0"),
            description="DialogRE: Human-annotated dialogue-based relation extraction dataset",
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
        self, filepath, split, max_seq_length=512, for_f1c=False, old_behaviour=False
    ):
        r"""Yields examples."""
        speaker_tokenizer = SpeakerBertTokenizer.from_pretrained("bert-base-uncased")
        with open(filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            if split == "train" and not for_f1c:
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
                            True,
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