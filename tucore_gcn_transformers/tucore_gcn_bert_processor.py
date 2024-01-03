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

r"""Processing code for DialogRE-formatted datasets

DialogRE is a human annotated dialogue-based relation extraction dataset. Existing DialogRE dataset loaders are insufficient for
project development. This module extends a pre-existing dataset loader from HuggingFace to add preprocessing capabilities for
model inputs

The following code is derived from external sources:
    The Google AI Language Team Authors and The HuggingFace Inc. team, Bongseok Lee:
        - [class] Conversation

    The HuggingFace Datasets Authors and the current dataset script contributor. [https://huggingface.co/datasets/dialog_re]
    [https://github.com/vineeths96]:
        - [class] DialogREConfig
        - [class] DialogRE

Original Works:
   - [class] SpeakerRelation
   - [class] Message
"""

import json
import os
from typing import Any, Dict, List, Union
import uuid
import datasets
import random
from dataclasses import dataclass
from datasets.utils.info_utils import VerificationMode
from datasets.utils.download_manager import DownloadManager
from tucore_gcn_bert_tokenizer import (
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


class SpeakerRelation:
    r"""Speaker Relation dataclass

    TUCORE-GCN implements a subject/object entity, labelled [unused1] and [unused2] in the official TUCORE-GCN repository,
    following {Dian Yu, Kai Sun, Claire Cardie, and Dong Yu. 2020, Dialogue-based relation extraction. In Proceedings of the
    58th Annual Meeting of the Association for Computational Linguistics}. Here, we have chosen to use entity_1 and entity_2 for
    better readability.

    DialogRE labels relations between entity_1 and entity_2 with a relations id. entities can have more than 1 relation at a
    time. Here, we store relation ids as a list

    Attributes:
        entity_1 (str):
            1st object/subject entity
        entity_2 (str):
            2st object/subject entity
        rid (List[int]):
            speaker input id totoken mapping
    """

    entity_1: str
    entity_2: str
    rid: List[int]

    def __init__(self, entity_1: str, entity_2: str, rid: List[int] = [37]):
        self.entity_1 = entity_1
        self.entity_2 = entity_2
        self.rid = rid

    def __repr__(self) -> str:
        return f"SpeakerRelation(entity_1=\"{self.entity_1}\",entity_2=\"{self.entity_2}\",rid={self.rid}"


@dataclass
class Message:
    speaker: str
    dialog: str

class Conversation:
    r"""Utility class containing a conversation between speakers, and relations between entities.

    Adapted from transformers library, transformers.pipelines.conversational.Conversation
    [https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/conversational.py] and from the official
    TUCORE-GCN repository [https://github.com/BlackNoodle/TUCORE-GCN]

    Describes a TUCORE-GCN compatible conversation. Manages a conversation repository, and provides utility functions for
    converting dialog-relation pairs to inputs for preprocessing.

    Arguments:
        messages (str|:obj:`list` of `str`|:obj:`list` of `Message`, *optional*):
                The initial messages to start the conversation, either a string, a list of strings, or a list of Message objects
                containing "speaker" and "dialog" attributes. All inputs are converted into the format `speaker`: `dialog`. Ex:
                `["Speaker 1: Hi!", "Speaker 2: Hey!"]`. NOTE: Speaker names must be labelled as "Speaker_N" for now.
        speaker_relations (:obj:`list` of `str`):
                A list of speaker relations between entities in the conversation.
        conversation_id (`uuid.UUID`, *optional*):
                Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the conversation.

    Modifications Summary:
        Adapted `is_speaker`, `rename`, `build_input_with_relation`, `build_inputs` from
        [https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py]. Modified Conversation constructor to be TUCORE-GCN input
        compatible

    Usage:

    ```python
    new_conversation = Conversation(
        messages=[
            Message("Speaker 1", "Howdy! I'm Flowey, Flowey the Flower!"), Message("Speaker 2", "Hello Flowey. I'm your very
            best friend!"),
        ], speaker_relations=[
            SpeakerRelation("Speaker 1", "Speaker 2")
        ]
    )
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
        r"""Generates a string representation of the conversation.

        Returns:
                `str`:

        Example:
                Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user: Going to the movies tonight - any suggestions?
                Relations: [SpeakerRelation(entity_1="speaker_1",entity_2="Alice",rid=[3])]
                speaker_1: Hello!
                speaker_2: Hello!!!!!
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
        self, relation, tokenizer, max_seq_length=512, for_f1c=False, old_behaviour=False
    ):
        r"""Builds TUCORE-GCN compatible inputs

        Adapted from https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py

        Converts conversation to stringified dialogs with entity names replaced with identifier tokens based on provided
        relation. The conversation is truncated to the last possible dialog where the total tokens in final model input would be
        less than or equal to max_seq_length.

        Arguments:
            relation (:obj:`SpeakerRelation`):
                    relation to stringify conversation with.
            tokenizer (:obj:`SpeakerBertTokenizer`):
                    Tokenizer used for checking total tokens
            max_seq_length (`uuid.UUID`, *optional*):
                    maxiumum tokens allowed in sequence. Inputs will be padded to this length

        Returns:
            dialog (`str`):
                Dtringified dialogs with entity names replaced with identifier tokens.
            relation (:obj:`SpeakerRelation`):
                Entity names in relation will additionally be replace with identifier tokens.

        Usage:

        ```python
        >>> ret_dialog, ret_relation = conversation.build_input_with_relation(relation, speaker_tokenizer).values()
        >>> ret_dialog
        "{entity_1}Hi!\n{speaker_2}Hello {entity_2}!"
        >>> ret_relation
        SpeakerRelation(entity_1="{entity_1}",entity_2="{entity_2}",rid=[3])
        ```
        """
        remove_colons = False
        dialog_raw = self.messages
        ret_relation = SpeakerRelation(
            relation.entity_1, relation.entity_2, relation.rid
        )
        soi = [
            SPEAKER_TOKENS.ENTITY_1,
            SPEAKER_TOKENS.ENTITY_2,
        ]
        ret_dialog = []
        a = []
        if self.is_speaker(relation.entity_1):
            a += [relation.entity_1]
        else:
            a += [None]
        if relation.entity_1 != relation.entity_2 and self.is_speaker(
            relation.entity_2
        ):
            a += [relation.entity_2]
        else:
            a += [None]
        for d in dialog_raw:
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", soi[i] + ("" if remove_colons else " :"))
            if not old_behaviour:
                d = d.replace(relation.entity_1, soi[0])
                d = d.replace(relation.entity_2, soi[1])
            d = d.replace("Speaker 1:", SPEAKER_TOKENS.SPEAKER_1 + ("" if remove_colons else " :"))
            d = d.replace("Speaker 2:", SPEAKER_TOKENS.SPEAKER_2 + ("" if remove_colons else " :"))
            d = d.replace("Speaker 3:", SPEAKER_TOKENS.SPEAKER_3 + ("" if remove_colons else " :"))
            d = d.replace("Speaker 4:", SPEAKER_TOKENS.SPEAKER_4 + ("" if remove_colons else " :"))
            d = d.replace("Speaker 5:", SPEAKER_TOKENS.SPEAKER_5 + ("" if remove_colons else " :"))
            d = d.replace("Speaker 6:", SPEAKER_TOKENS.SPEAKER_6 + ("" if remove_colons else " :"))
            d = d.replace("Speaker 7:", SPEAKER_TOKENS.SPEAKER_7 + ("" if remove_colons else " :"))
            d = d.replace("Speaker 8:", SPEAKER_TOKENS.SPEAKER_8 + ("" if remove_colons else " :"))
            d = d.replace("Speaker 9:", SPEAKER_TOKENS.SPEAKER_9 + ("" if remove_colons else " :"))
            ret_dialog.append(d)
        if old_behaviour:
            for i in range(len(a)):
                if a[i] is None:
                    continue
                if relation.entity_1 == a[i]:
                    ret_relation.entity_1 = soi[i]
                if relation.entity_2 == a[i]:
                    ret_relation.entity_2 = soi[i]
        else:
            ret_relation.entity_1 = soi[0]
            ret_relation.entity_2 = soi[1]
        dialog, relation = ret_dialog, ret_relation
        count = 0
        lim_dialog = []
        speaker_x_tokens = tokenizer.tokenize(relation.entity_1)
        speaker_y_tokens = tokenizer.tokenize(relation.entity_2)
        for line in dialog:
            line_len = len(tokenizer.tokenize(line))
            if (
                count + line_len + len(speaker_x_tokens) + len(speaker_y_tokens) + 4
                > max_seq_length
            ):
                break
            count += line_len
            lim_dialog.append(line)
        return {
            "dialog": "\n".join(lim_dialog).lower(),
            "relation": {"entity_1": relation.entity_1, "entity_2": relation.entity_2, "rid": relation.rid},
        }

    def build_inputs(self, tokenizer, max_seq_length=512, for_f1c=False, old_behaviour=False):
        r"""Builds TUCORE-GCN compatible inputs

        Adapted from https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py

        Runs build_input_with_relation for all relations to obtain a list of stringified dialogs.

        Arguments:
            tokenizer (:obj:`SpeakerBertTokenizer`):
                    Tokenizer used for checking total tokens
            max_seq_length (`uuid.UUID`, *optional*):
                    maxiumum tokens allowed in sequence. Inputs will be padded to this length

        Returns:
            List of dialog, relation from build_input_with_relation

        Usage:

        ```python
        >>> dialogs = conversation.build_input_with_relation(relation, speaker_tokenizer).values()
        >>> dialogs[0]['dialog']
        "{entity_1}Hi!\n{speaker_2}Hello {entity_2}!"
        >>> dialogs[0]['relation']
        SpeakerRelation(entity_1="{entity_1}",entity_2="{entity_2}",rid=[3])
        ```
        """
        ret = []
        speaker_relations_iterator = enumerate(self.speaker_relations)
        while True:
            idx, relation = next(speaker_relations_iterator, (-1, -1))
            if idx == -1:
                break
            ret_dialog, ret_relation = self.build_input_with_relation(
                relation, tokenizer, max_seq_length, for_f1c, old_behaviour
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
    r"""BuilderConfig for DialogRE"""

    def __init__(self, **kwargs):
        r"""BuilderConfig for DialogRE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DialogREConfig, self).__init__(**kwargs)


class DialogRE(datasets.GeneratorBasedBuilder):
    r"""DialogRE: Human-annotated dialogue-based relation extraction dataset Version 2

    Adapted from https://huggingface.co/datasets/dialog_re, https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py

    Dataset Loader for DialogRE

    Modifications Summary:
        Added preprocessing to _generate_examples.
    """

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
                    "dialog": datasets.Value("string"),
                    "relation": datasets.Features(
                        {
                            "entity_1": datasets.Value("string"),
                            "entity_2": datasets.Value("string"),
                            "rid": datasets.Sequence(datasets.Value("int32")),
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
                        relation, speaker_tokenizer, max_seq_length, for_f1c, old_behaviour
                    ).values()
                    if ret_dialog != "":
                        yield idx, {
                            "dialog": ret_dialog,
                            "relation": ret_relation,
                        }
