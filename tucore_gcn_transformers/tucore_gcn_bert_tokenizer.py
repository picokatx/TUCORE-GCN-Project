# coding=utf-8

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

"""
Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
"""
* The following code is derived from The Google AI Language Team Authors and The
* HuggingFace Inc. team.
*    [class] SpeakerBertTokenizer
* Original Works 
*    [dataclass] SPEAKER_TOKENS
* 
"""

from dataclasses import dataclass
from transformers.models.bert.tokenization_bert import BertTokenizer

"""
* **SpeakerBertTokenizer**
* Original Work
* 
* Simple dataclass that handles all speaker token to id mapping. 
* 
"""


@dataclass
class SPEAKER_TOKENS:
    SPEAKER_1 = "{speaker_1}"
    SPEAKER_2 = "{speaker_2}"
    SPEAKER_3 = "{speaker_3}"
    SPEAKER_4 = "{speaker_4}"
    SPEAKER_5 = "{speaker_5}"
    SPEAKER_6 = "{speaker_6}"
    SPEAKER_7 = "{speaker_7}"
    SPEAKER_8 = "{speaker_8}"
    SPEAKER_9 = "{speaker_9}"
    SPEAKER_X = "{speaker_x}"
    SPEAKER_Y = "{speaker_y}"

    def is_speaker(token):
        speaker_tokens = [
            entry[1]
            for idx, entry in list(
                filter(
                    lambda x: x[1][1] if not x[1][0].startswith("__") else None,
                    enumerate(SPEAKER_TOKENS.__dict__.items()),
                )
            )
        ]
        return speaker_tokens.count(token) != 0

    def convert_speaker_to_id(token):
        speaker_tokens = [
            entry[1]
            for idx, entry in list(
                filter(
                    lambda x: x[1][1] if not x[1][0].startswith("__") else None,
                    enumerate(SPEAKER_TOKENS.__dict__.items()),
                )
            )
        ]
        return speaker_tokens.index(token)


"""
* **SpeakerBertTokenizer**
* Adapted from transformers library, transformers.models.bert.tokenization_bert.BertTokenizer
* [https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py]
* 
* A Bert Tokenizer.
* 
* **Modifications Summary**
* Added `speaker2id` and `id2speaker` for mapping speaker tokens to the ids used
* in the official TUCORE-GCN repository. Modified class methods `_tokenize`,
* `_convert_token_to_id`, `_convert_id_to_token` to treat speaker tokens as
* special tokens, that cannot be split on and have predefined ids. Due to the
* original work defining only 9 Speaker tokens, and 2 speakers of interest, we
* do the same here
* 
* **Methods**
* [func] `__init__`
*    vocab_file[TUCOREGCN_BertConfig]: Pass in `vocab.txt` packaged with pre-trained BERT from
*    HuggingFace
* [func] `_tokenize`
*    Uses same args as BertTokenizer
*    Note that `never_split` is deprecated, but still has functionality and is
*    used for special behaviour with speaker tokens
* [func] `_convert_token_to_id`,`_convert_id_to_token`
*    Uses same args as BertTokenizer
*    ids/tokens are first checked for speakers before anything else is done.
"""


class SpeakerBertTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            do_lower_case,
            do_basic_tokenize,
            never_split,
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
            tokenize_chinese_chars,
            strip_accents,
            **kwargs,
        )
        self.speaker2id = {
            SPEAKER_TOKENS.SPEAKER_X: 11,
            SPEAKER_TOKENS.SPEAKER_Y: 12,
            SPEAKER_TOKENS.SPEAKER_1: 1,
            SPEAKER_TOKENS.SPEAKER_2: 2,
            SPEAKER_TOKENS.SPEAKER_3: 3,
            SPEAKER_TOKENS.SPEAKER_4: 4,
            SPEAKER_TOKENS.SPEAKER_5: 5,
            SPEAKER_TOKENS.SPEAKER_6: 6,
            SPEAKER_TOKENS.SPEAKER_7: 7,
            SPEAKER_TOKENS.SPEAKER_8: 8,
            SPEAKER_TOKENS.SPEAKER_9: 9,
        }

        self.id2speaker = {
            "11": SPEAKER_TOKENS.SPEAKER_X,
            "12": SPEAKER_TOKENS.SPEAKER_Y,
            "1": SPEAKER_TOKENS.SPEAKER_1,
            "2": SPEAKER_TOKENS.SPEAKER_2,
            "3": SPEAKER_TOKENS.SPEAKER_3,
            "4": SPEAKER_TOKENS.SPEAKER_4,
            "5": SPEAKER_TOKENS.SPEAKER_5,
            "6": SPEAKER_TOKENS.SPEAKER_6,
            "7": SPEAKER_TOKENS.SPEAKER_7,
            "8": SPEAKER_TOKENS.SPEAKER_8,
            "9": SPEAKER_TOKENS.SPEAKER_9,
        }
        self.basic_tokenizer.never_split = set(
            self.all_special_tokens + list(self.speaker2id.keys())
        )

    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                text,
                never_split=set(self.all_special_tokens + list(self.speaker2id.keys())),
            ):
                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.speaker2id:
            return self.speaker2id[token]
        else:
            return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if str(index) in self.id2speaker:
            return self.speaker2id[str(index)]
        else:
            return self.ids_to_tokens.get(index, self.unk_token)
