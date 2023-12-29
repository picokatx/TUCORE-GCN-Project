# coding=utf-8

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

# Dian Yu, Kai Sun, Claire Cardie, and Dong Yu. 2020. Dialogue-based relation
# extraction. In Proceedings of the 58th Annual Meeting of the Association for
# Computational Linguistics, pages 4927–4940, Online. Association for
# Computational Linguistics.
# https://github.com/nlpdata/dialogre
# dataset is intended for non-commercial research purpose only.

# MIT License
# 
# Copyright (c) 2021 Bongseok Lee
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# MIT License
# 
# Copyright (c) 2024 picokatx
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Processing code for DialogRE-formatted datasets

The following code is derived from The Google AI Language Team Authors and The
HuggingFace Inc. team, Bongseok Lee
   [class] Conversation
The following code is derived from The HuggingFace Datasets Authors and the
current dataset script contributor. The contributors can be found here:
[https://huggingface.co/datasets/dialog_re] [https://github.com/vineeths96]
   [class] DialogREConfig
   [class] DialogRE
Original Works 
   [class] SpeakerRelation
   [dataclass] Message


"""

from dataclasses import dataclass
from transformers.models.bert.tokenization_bert import BertTokenizer

"""DialogRE: the first human-annotated dialogue-based relation extraction dataset"""

import json
import os
from typing import Any, Dict, List, Union
import uuid
import datasets
import random
from dataclasses import dataclass
from datasets.utils.info_utils import VerificationMode
from datasets.utils.download_manager import DownloadManager
from tucore_gcn_transformers.tucore_gcn_bert_tokenizer import (
    SPEAKER_TOKENS,
    SpeakerBertTokenizer,
)
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

"""
* **SpeakerRelation**
* Original Work
* 
* dataclass containing speaker pairs and a relation id, as described by DialogRE
* 
"""

class SpeakerRelation:
    speaker_x: str
    speaker_y: str
    rid: List[int]

    def __init__(self, speaker_x:str, speaker_y:str, rid:List[int]=[37]) -> None:
        self.speaker_x = speaker_x
        self.speaker_y = speaker_y
        self.rid = rid

"""
* **Message**
* Original Work
* 
* dataclass containing speaker dialog mappings
* 
"""

@dataclass
class Message:
    speaker: str
    dialog: str

"""
* **Conversation**
* Adapted from transformers library, transformers.pipelines.conversational.Conversation
* [https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/conversational.py]
* and from the official TUCORE-GCN repository
* [https://github.com/BlackNoodle/TUCORE-GCN]
* Describes a TUCORE-GCN compatible conversation. Manages a conversation
* repository, and provides utility functions for converting dialog-relation
* pairs to inputs for preprocessing.
* 
* **Modifications Summary**
* Adapted `is_speaker`, `rename`, `build_input_with_relation`, `build_inputs`
* from [https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py] 
* 
* **Methods**
* [func] `__init__`
*    messages (Union[str, List[str], List[Message]]): Messages comprising the conversation. Converts any input to the
*    format `speaker`: `message`. Ex: `["Speaker 1: Hi!", "Speaker 2: Hi!"]`
* [func] `is_speaker`
*    Checks if input string s has the formatting `speaker X` where X is any digit
* [func] `rename` 
*    dialog (List[str]): 
*    relation (SpeakerRelation): 
* [func] `_convert_token_to_id`,`_convert_id_to_token`
* Usage:
* ```js
  new_conversation = Conversation(
      messages=[
          Message("Speaker 1", "Howdy! I'm Flowey, Flowey the Flower!"),
          Message("Speaker 2", "Hello Flowey. I'm your very best friend!"),
      ],
      speaker_relations=[
          SpeakerRelation("Speaker 1", "Speaker 2")
      ]
  )
* ```
"""

class Conversation:
    """
    Utility class containing a conversation and its history. This class is meant to be used as an input to the
    [`ConversationalPipeline`]. The conversation contains several utility functions to manage the addition of new user
    inputs and generated model responses.

    Arguments:
            messages (Union[str, List[Dict[str, str]]], *optional*):
                    The initial messages to start the conversation, either a string, or a list of dicts containing "role" and
                    "content" keys. If a string is passed, it is interpreted as a single message with the "user" role.
            conversation_id (`uuid.UUID`, *optional*):
                    Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
                    conversation.

    Usage:

    ```python
    conversation = Conversation("Going to the movies tonight - any suggestions?")
    conversation.add_message({"role": "assistant", "content": "The Big lebowski."})
    conversation.add_message({"role": "user", "content": "Is it good?"})
    ```
    """

    def __init__(
        self,
        messages: Union[str, List[str], List[Message]] = [],
        speaker_relations: List[SpeakerRelation] = [],
        conversation_id: uuid.UUID = None,
        **deprecated_kwargs,
    ):
        if not conversation_id:
            conversation_id = uuid.uuid4()

        if isinstance(messages, List):
            if isinstance(messages[0], Message):
                messages = [
                    message.speaker + ": " + message.dialog for message in messages
                ]
        elif isinstance(messages, str):
            messages = [messages]
        self.uuid = conversation_id
        self.speaker_relations = speaker_relations
        self.messages = messages

    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        return self.uuid == other.uuid or self.messages == other.messages

    def add_message(self, message: Message):
        self.messages.append(message)

    def __iter__(self):
        for message in self.messages:
            yield message

    def __getitem__(self, item):
        return self.messages[item]

    def __setitem__(self, key, value):
        self.messages[key] = value

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        """
        Generates a string representation of the conversation.

        Returns:
                `str`:

        Example:
                Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user: Going to the movies tonight - any suggestions?
                bot: The Big Lebowski
        """
        output = f"Conversation id: {self.uuid}\n"
        output += f"Relations: {self.speaker_relations}\n"
        for message in self.messages:
            output += f"{message}\n"
        return output

    def is_speaker(self, s: str):
        s = s.split()
        return len(s) == 2 and s[0] == "Speaker" and s[1].isdigit()

    def build_input_with_relation(
        self, relation, tokenizer, max_length=512, for_f1c=False
    ):
        '''
        if for_f1c:
            for l in range(1, len(self.messages)+1):
            dialog, relation = self.rename(self.messages[:l], relation)
        else:
            dialog, relation = self.rename(self.messages, relation
        '''
        dialog_raw = self.messages
        ret_relation = SpeakerRelation(
            relation.speaker_x, relation.speaker_y, relation.rid
        )
        soi = [
            SPEAKER_TOKENS.SPEAKER_X,
            SPEAKER_TOKENS.SPEAKER_Y,
        ]  # speaker_of_interest
        ret_dialog = []
        a = []
        if self.is_speaker(relation.speaker_x):
            a += [relation.speaker_x]
        else:
            a += [None]
        if relation.speaker_x != relation.speaker_y and self.is_speaker(
            relation.speaker_y
        ):
            a += [relation.speaker_y]
        else:
            a += [None]
        for d in dialog_raw:
            for i in range(len(a)):
                if a[i] is None:
                    continue
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
        for i in range(len(a)):
            if a[i] is None:
                continue
            if relation.speaker_x == a[i]:
                ret_relation.speaker_x = soi[i]
            if relation.speaker_y == a[i]:
                ret_relation.speaker_y = soi[i]
        dialog, relation = ret_dialog, ret_relation
        count = 0
        lim_dialog = []
        speaker_x_tokens = tokenizer.tokenize(relation.speaker_x)
        speaker_y_tokens = tokenizer.tokenize(relation.speaker_y)
        for line in dialog:
            line_len = len(tokenizer.tokenize(line))
            if (
                count + line_len + len(speaker_x_tokens) + len(speaker_y_tokens) + 4
                > max_length
            ):
                break
            count += line_len
            lim_dialog.append(line)
        return {
            "dialog": "\n".join(lim_dialog).lower(),
            "relation": relation,
        }

    def build_inputs(self, tokenizer, max_length=512, for_f1c=False):
        ret = []
        speaker_relations_iterator = enumerate(self.speaker_relations)
        while True:
            idx, relation = next(speaker_relations_iterator, (-1, -1))
            if idx == -1:
                break
            ret_dialog, ret_relation = self.build_input_with_relation(
                relation, tokenizer, max_length, for_f1c
            ).values()
            if ret_dialog != "":
                ret.append(
                    {
                        "dialog": ret_dialog,
                        "relation": ret_relation,
                    }
                )
        return ret


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
            verification_mode,
            check_duplicate_keys=verification_mode == VerificationMode.BASIC_CHECKS
            or verification_mode == VerificationMode.ALL_CHECKS,
            **prepare_splits_kwargs,
        )

    # _get_examples_iterable_for_split
    def _generate_examples(self, filepath, split, max_length=512, for_f1c=False):
        """Yields examples."""
        speaker_tokenizer = SpeakerBertTokenizer.from_pretrained("bert-base-uncased")
        with open(filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            if split == "train" and not for_f1c:
                random.shuffle(dataset)
            for idx, entry in enumerate(dataset):
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
                        relation, speaker_tokenizer, max_length, for_f1c
                    ).values()
                    if ret_dialog != "":
                        yield idx, {
                            "dialog": ret_dialog,
                            "relation": ret_relation,
                        }
