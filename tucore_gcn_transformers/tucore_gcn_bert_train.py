import os
from optimization import BERTAdam
from tucore_gcn_transformers.tucore_gcn_bert_tokenizer import SpeakerBertTokenizer
from tucore_gcn_transformers.tucore_gcn_bert_pipeline import create_model_inputs
from tucore_gcn_transformers.tucore_gcn_bert_modelling import TUCOREGCN_BertConfig, TUCOREGCN_BertForSequenceClassification
from tqdm.notebook import tqdm_notebook
import json
from tucore_gcn_transformers.tucore_gcn_bert_processor import (
    SpeakerRelation,
    Conversation,
    DialogRE
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

def build_inputs_from_dialogre():
    dialogre = DialogRE()
    dialogre.download_and_prepare()
    dialogre_data = dialogre.as_dataset()
    speaker_tokenizer = SpeakerBertTokenizer.from_pretrained("bert-base-uncased")
    return {
        "train": [
            create_model_inputs(
                speaker_tokenizer.tokenize(entry["dialog"]),
                speaker_tokenizer.tokenize(entry["relation"]["entity_1"]),
                speaker_tokenizer.tokenize(entry["relation"]["entity_2"]),
                speaker_tokenizer,
                entry,
                True,
                36,
                512,
            )
            for idx, entry in tqdm(
                enumerate(dialogre_data["train"]), total=dialogre_data["train"].num_rows
            )
        ],
        "test": [
            create_model_inputs(
                speaker_tokenizer.tokenize(entry["dialog"]),
                speaker_tokenizer.tokenize(entry["relation"]["entity_1"]),
                speaker_tokenizer.tokenize(entry["relation"]["entity_2"]),
                speaker_tokenizer,
                entry,
                True,
                36,
                512,
            )
            for idx, entry in tqdm(
                enumerate(dialogre_data["test"]), total=dialogre_data["test"].num_rows
            )
        ],
        "dev": [
            create_model_inputs(
                speaker_tokenizer.tokenize(entry["dialog"]),
                speaker_tokenizer.tokenize(entry["relation"]["entity_1"]),
                speaker_tokenizer.tokenize(entry["relation"]["entity_2"]),
                speaker_tokenizer,
                entry,
                True,
                36,
                512,
            )
            for idx, entry in tqdm(
                enumerate(dialogre_data["validation"]),
                total=dialogre_data["validation"].num_rows,
            )
        ],
    }


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
            args.max_seq_length, config.max_position_embeddings))
    # Don't overwrite last session's test model
    if os.path.exists(args.output_dir) and 'model.pt' in os.listdir(args.output_dir):
        if args.do_train and not args.resume:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    # Custom speaker tokenizer with using deprecated workaround but who cares
    tokenizer = SpeakerBertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    data = build_inputs_from_dialogre()
    train_set = data['train']
    test_set = data['test']
    num_train_steps = int(len(train_set) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    model = TUCOREGCN_BertForSequenceClassification(config, n_class)
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'), strict=False)
    no_decay = ['bias', 'gamma', 'beta']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
        ]
    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    global_step = 0
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    return



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

    '''
    "tokens": tokens,
    "input_ids": input_ids,
    "input_mask": input_mask,
    "segment_ids": segment_ids,
    "speaker_ids": speaker_ids,
    "mention_ids": mention_ids,
    "turn_masks": turn_masks,
    "graph": graph,
    '''

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "input_ids": datasets.Sequence(datasets.Value("int32")),
                    "input_mask": datasets.Sequence(datasets.Value("int32")),
                    "segment_ids": datasets.Sequence(datasets.Value("int32")),
                    "speaker_ids": datasets.Sequence(datasets.Value("int32")),
                    "mention_ids": datasets.Sequence(datasets.Value("int32")),
                    "turn_masks": datasets.Sequence(datasets.Sequence(datasets.Value("bool"))),
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

    def _generate_examples(self, filepath, split, max_seq_length=512, for_f1c=False, old_behaviour=False):
        r"""Yields examples."""
        speaker_tokenizer = SpeakerBertTokenizer.from_pretrained("bert-base-uncased")
        with open(filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            if split == "train" and not for_f1c:
                random.shuffle(dataset)
            for tqdm_idx, entry in tqdm_notebook(enumerate(dataset), total=len(dataset)):
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
                        relation, speaker_tokenizer, max_seq_length, for_f1c, old_behaviour
                    ).values()
                    if ret_dialog != "":
                        entry = {
                            "dialog": ret_dialog,
                            "relation": ret_relation,
                        }
                        (
                            tokens,
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
                        )
                        '''if tqdm_idx==184:
                            print({
                                "tokens": tokens,
                                "input_ids": input_ids,
                                "input_mask": input_mask,
                                "segment_ids": segment_ids,
                                "speaker_ids": speaker_ids,
                                "mention_ids": mention_ids,
                                "turn_masks": turn_masks,
                                "graph": pickle.dumps(graph),
                            })'''
                        yield idx, {
                            "tokens": tokens[0],
                            "input_ids": input_ids[0],
                            "input_mask": input_mask[0],
                            "segment_ids": segment_ids[0],
                            "speaker_ids": speaker_ids[0],
                            "mention_ids": mention_ids[0],
                            "turn_masks": turn_masks[0],
                            "graph": pickle.dumps(graph),
                        }

