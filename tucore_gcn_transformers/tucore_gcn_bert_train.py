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
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple, Union
import random
from datasets.utils.info_utils import VerificationMode
from datasets.data_files import DataFilesDict
from datasets.info import DatasetInfo
from datasets.features import Features
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

"""
python tucore_gcn_transformers/tucore_gcn_bert_train.py --do_train --do_eval --output_dir tucore_gcn_bert_test1
python tucore_gcn_transformers/tucore_gcn_bert_train.py --do_train --do_eval --output_dir tucore_gcn_bert_test1 --num_train_epochs 1
python tucore_gcn_transformers/tucore_gcn_bert_train.py --do_train --do_eval --output_dir tucore_gcn_bert_test1 --num_train_epochs 1 --resume [X]
python tucore_gcn_transformers/tucore_gcn_bert_train.py --do_train --do_eval --do_test_eval 1 --output_dir TUCOREGCN_BERT_DialogRE --num_train_epochs 1 --resume 69 --eval_batch_size 32
4.0gb
2.9gb
seq_length of 512
"""

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

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config_name: Optional[str] = None,
        hash: Optional[str] = None,
        base_path: Optional[str] = None,
        info: Optional[DatasetInfo] = None,
        features: Optional[Features] = None,
        token: Optional[Union[bool, str]] = None,
        use_auth_token="deprecated",
        repo_id: Optional[str] = None,
        data_files: Optional[Union[str, list, dict, DataFilesDict]] = None,
        data_dir: Optional[str] = None,
        storage_options: Optional[dict] = None,
        writer_batch_size: Optional[int] = None,
        name="deprecated",
        max_seq_length=512,
        old_behaviour=True,
        model_type="bert",
        **config_kwargs,
    ):
        super().__init__(
            cache_dir,
            dataset_name,
            config_name,
            hash,
            base_path,
            info,
            features,
            token,
            use_auth_token,
            repo_id,
            data_files,
            data_dir,
            storage_options,
            writer_batch_size,
            name,
            **config_kwargs,
        )
        self.max_seq_length = max_seq_length
        self.old_behaviour = old_behaviour
        self.model_type = model_type

    # override init to pass additional args
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
                    "graph_data": datasets.Value("binary"),
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
                    "max_seq_length": self.max_seq_length,
                    "old_behaviour": self.old_behaviour,
                    "shuffle": True,
                    "model_type": self.model_type,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["test"]),
                    "split": "test",
                    "max_seq_length": self.max_seq_length,
                    "old_behaviour": self.old_behaviour,
                    "shuffle": False,
                    "model_type": self.model_type,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["dev"]),
                    "split": "dev",
                    "max_seq_length": self.max_seq_length,
                    "old_behaviour": self.old_behaviour,
                    "shuffle": False,
                    "model_type": self.model_type,
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
        self, filepath, split, max_seq_length, old_behaviour, shuffle, model_type
    ):
        r"""Yields examples."""
        speaker_tokenizer: SpeakerBertTokenizer = SpeakerBertTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        # train has accents, while test and dev is already normalized.
        speaker_tokenizer.basic_tokenizer.strip_accents = split == "train"
        with open(filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            if split == "train" and shuffle:
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
                            graph_data,
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
                            True,
                            model_type=model_type,
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
                            "graph_data": pickle.dumps(graph_data),
                            #    "graph_speaker": graph_data[0][('node', 'speaker', 'node')],
                            #    "graph_dialog": graph_data[0][('node', 'dialog', 'node')],
                            #    "graph_entity": graph_data[0][('node', 'entity', 'node')],
                        }


if __name__ == "__main__":
    # main()
    pass
