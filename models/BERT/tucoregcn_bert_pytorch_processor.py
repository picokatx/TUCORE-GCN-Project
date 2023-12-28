# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DialogRE: the first human-annotated dialogue-based relation extraction dataset"""

import json
import os
from typing import List
import datasets
import random
from dataclasses import dataclass
from datasets.utils.info_utils import VerificationMode
from datasets.utils.download_manager import DownloadManager
from models.BERT.speaker_tokens import SPEAKER_TOKENS, SpeakerBertTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
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

@dataclass
class DialogRERelation():
	speaker_x: str
	speaker_y: str
	rid: List[int] #one hot
	def from_dialogRE(entry, n_class):
		rid = []
		for k in range(n_class):
			if k+1 in entry['rid']:
				rid += [1]
			else:
				rid += [0]
		return DialogRERelation(entry['x'], entry['y'], rid)

class DialogREConfig(datasets.BuilderConfig):
	"""BuilderConfig for DialogRE"""

	def __init__(self, **kwargs):
		"""BuilderConfig for DialogRE.
		Args:
		  **kwargs: keyword arguments forwarded to super.
		"""
		super(DialogREConfig, self).__init__(**kwargs)


class DialogRE(datasets.GeneratorBasedBuilder):
	"""DialogRE: Human-annotated dialogue-based relation extraction dataset Version 2"""

	VERSION = datasets.Version("1.1.0")

	BUILDER_CONFIGS = [
		DialogREConfig(
			name="dialog_re",
			version=datasets.Version("1.1.0"),
			description="DialogRE: Human-annotated dialogue-based relation extraction dataset",
		),
	]
	
	def _info(self):
		return datasets.DatasetInfo(
			description=_DESCRIPTION,
			features=datasets.Features(
				{
					"dialog": datasets.Sequence(datasets.Value("string")),
					"relation_data": datasets.Sequence(
						{
							"x": datasets.Value("string"),
							"y": datasets.Value("string"),
							"x_type": datasets.Value("string"),
							"y_type": datasets.Value("string"),
							"r": datasets.Sequence(datasets.Value("string")),
							"rid": datasets.Sequence(datasets.Value("int32")),
							"t": datasets.Sequence(datasets.Value("string")),
						}
					),
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
				gen_kwargs={"filepath": os.path.join(data_dir["test"]), "split": "test"},
			),
			datasets.SplitGenerator(
				name=datasets.Split.VALIDATION,
				gen_kwargs={
					"filepath": os.path.join(data_dir["dev"]),
					"split": "dev",
				},
			),
		]
	def _download_and_prepare(self, dl_manager, verification_mode, **prepare_splits_kwargs):
		super()._download_and_prepare(
			dl_manager,
			verification_mode,
			check_duplicate_keys=verification_mode == VerificationMode.BASIC_CHECKS
			or verification_mode == VerificationMode.ALL_CHECKS,
			**prepare_splits_kwargs,
		)
	
	#_get_examples_iterable_for_split
	def _generate_examples(self, filepath, split, n_class=36, max_length=512, for_f1c=False):
		def is_speaker(a):
				a = a.split()
				return len(a) == 2 and a[0] == "Speaker" and a[1].isdigit()
		def rename(dialog: List[str], relation: DialogRERelation):
				soi = [SPEAKER_TOKENS.SPEAKER_X, SPEAKER_TOKENS.SPEAKER_Y] #speaker_of_interest
				ret_dialog = []
				a = []
				if is_speaker(relation.speaker_x):
					a += [relation.speaker_x]
				else:
					a += [None]
				if relation.speaker_x != relation.speaker_y and is_speaker(relation.speaker_y):
					a += [relation.speaker_y]
				else:
					a += [None]
				for i in range(len(a)):
					if a[i] is None:
						continue
					for d in dialog:
						d = d.replace(a[i] + ":", soi[i])
						d = d.replace("Speaker 1:", SPEAKER_TOKENS.SPEAKER_1)
						d = d.replace("Speaker 2:", SPEAKER_TOKENS.SPEAKER_2)
						d = d.replace("Speaker 3:", SPEAKER_TOKENS.SPEAKER_3)
						d = d.replace("Speaker 4:", SPEAKER_TOKENS.SPEAKER_4)
						d = d.replace("Speaker 5:", SPEAKER_TOKENS.SPEAKER_5)
						d = d.replace("Speaker 6:", SPEAKER_TOKENS.SPEAKER_6)
						d = d.replace("Speaker 7:", SPEAKER_TOKENS.SPEAKER_7)
						d = d.replace("Speaker 8:", SPEAKER_TOKENS.SPEAKER_8)
						d = d.replace("Speaker 9:", SPEAKER_TOKENS.SPEAKER_9)
						ret_dialog.append(d)
					if relation.speaker_x == a[i]:
						relation.speaker_x = soi[i]
					if relation.speaker_y == a[i]:
						relation.speaker_y = soi[i]
				return ret_dialog, relation
		"""Yields examples."""
		speaker_tokenizer = SpeakerBertTokenizer.from_pretrained("bert-base-uncased")
		with open(filepath, encoding="utf-8") as f:
			dataset = json.load(f)
			if split=="train" and not for_f1c:
				random.shuffle(dataset)
			for idx, entry in enumerate(dataset):
				dialog_raw = entry[0]
				relation_data = entry[1]
				relations = [DialogRERelation.from_dialogRE(relation, n_class) for relation in relation_data]
				for idx, relation in enumerate(relations):
					if for_f1c:
						for l in range(1, len(dialog_raw)+1):
							dialog, relation = rename(dialog_raw[:l], relation)
					else:
						dialog, relation = rename(dialog_raw, relation)
					count = 0
					lim_dialog = []
					speaker_x_tokens = speaker_tokenizer.tokenize(relation.speaker_x)
					speaker_y_tokens = speaker_tokenizer.tokenize(relation.speaker_y)
					for line in dialog:
						line_len = len(speaker_tokenizer.tokenize(line))
						if count+line_len+len(speaker_x_tokens)+len(speaker_y_tokens)+4>max_length: break
						count+=line_len
						lim_dialog.append(line)
					yield idx, {
						"dialog": "\n".join(lim_dialog).lower(),
						"relation": relation,
					}

