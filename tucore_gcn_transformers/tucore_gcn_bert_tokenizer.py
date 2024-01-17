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

r"""Tokenization classes for TUCORE-GCN

transformer's BERTTokenizer is not sufficient for handling conversation inputs. This module extends BERTTokenizer's capabilities
to include special tokenizing behavior when handling speaker tokens.

The following are is derived from external sources:
    The Google AI Language Team Authors and The HuggingFace Inc. team:
        - [class] SpeakerBertTokenizer

Original Works:
   - [dataclass] SPEAKER_TOKENS

NOTE: Speaker Id as is used here should not be confused with speaker_id. Speaker Id is passed as an input_ids model parameter,
while speaker_ids is a different parameter and has different token2id mappings
"""

from dataclasses import dataclass
from transformers.models.bert.tokenization_bert import BertTokenizer


@dataclass
class SPEAKER_TOKENS:
    r"""Mappings for TUCORE-GCN's speaker2token

    TUCORE-GCN implements Speakers's 1-9, and a subject and object speaker, labelled [unused1] and [unused2] in the official
    repository. Here, we have chosen to use speaker_x and speaker_y for better readability.

    Attributes:
        SPEAKER_N (str): speaker input id to token mapping
    """

    SPEAKER_1 = "{speaker_1}"
    SPEAKER_2 = "{speaker_2}"
    SPEAKER_3 = "{speaker_3}"
    SPEAKER_4 = "{speaker_4}"
    SPEAKER_5 = "{speaker_5}"
    SPEAKER_6 = "{speaker_6}"
    SPEAKER_7 = "{speaker_7}"
    SPEAKER_8 = "{speaker_8}"
    SPEAKER_9 = "{speaker_9}"
    ENTITY_1 = "{entity_1}"
    ENTITY_2 = "{entity_2}"


class SpeakerBertTokenizer(BertTokenizer):
    r"""BERT Tokenizer with speaker-id mappings. Based on WordPiece.

    Adapted from transformers library, transformers.models.bert.tokenization_bert.BertTokenizer
    [https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py]

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Added speaker2id to self.basic_tokenizer.never_split, preventing BertTokenizer's internal tokenizer from splitting speaker
    tokens.

    Modifications Summary:
        Added speaker2id and id2speaker for mapping speaker tokens to the ids use in the official TUCORE-GCN repository.
        Modified class methods _tokenize, _convert_token_to_id, _convert_id_to_token to treat speaker tokens as special tokens,
        that cannot be split on and have predefined ids. Due to the original work defining only 9 Speaker tokens, and 2 speakers
        of interest, we do the same here

    Added/Modified Attributes:
        speaker2id (:obj:`dict` of `str`,`int`):
            speaker2id as specified by TUCORE-GCN
        id2speaker (:obj:`dict` of `int`,`str`):
            id2speaker as specified by TUCORE-GCN

    Added/Modified Methods:
        __init__, _tokenize, is_speaker, convert_speaker_to_id, _convert_token_to_id, _convert_id_to_token

    Arguments:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token
            instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence
            classification or for a text and a question for question answering. It is also used as the last token of a sequence
            built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence instead
            of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language modeling.
            This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the value for
            `lowercase` (as in the original BERT).

    Usage:

    ```python
    >>> tokenizer = SpeakerBertTokenizer.from_pretrained('bert-base-uncased')
    >>> tokenizer.tokenize("speaker_1: lorem, ipsum docet?")
    ['{speaker_1}', 'lorem', ',', 'ipsum', 'docet', '?']
    ```

    NOTE: self.basic_tokenizer.never_split is deprecated, but still has functionality
    WARNING: If a name is not tokenizable, it is replaced with [UNK]. This issue is present in the original TUCORE-GCN
    repository, and it leads to inaccurate entity-turn edges being formed.
    e.g. 
    "x": "Speaker 1",
    "y": "Pheebs",
    "rid": [
     30
    ],
    "pheebs" is not a registered token or partial token in vocab.txt. it will tokenize to [UNK] (id=100).
    NOTE: The benefits offered by BertTokenizer mean some parity with the data processing methods used by TUCORE-GCN is lost. In
    particular, words with accents are converted to their vocab.txt representations
    """

    speaker2id = {
        SPEAKER_TOKENS.ENTITY_1: 11,
        SPEAKER_TOKENS.ENTITY_2: 12,
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

    id2speaker = {
        "11": SPEAKER_TOKENS.ENTITY_1,
        "12": SPEAKER_TOKENS.ENTITY_2,
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

    def __init__(
        self,
        vocab_file: str,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents: bool = None,
        old_behaviour = True,
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
        self.old_behaviour = old_behaviour
        self.basic_tokenizer.never_split = set(
            self.all_special_tokens + (list(self.speaker2id.keys()) if not old_behaviour else [SPEAKER_TOKENS.ENTITY_1, SPEAKER_TOKENS.ENTITY_2])
        )
    def _tokenize(self, text: str, split_special_tokens=False):
        r"""BertTokenizer's internal tokenize method

        Added speaker2id to self.basic_tokenizer.never_split, preventing BertTokenizer's internal tokenizer from splitting
        speaker tokens.

        Arguments:
            text (`str`):
                Text to be tokenized
            split_special_tokens (`bool`):
                Whether whitespace tokenizing should be performed on special tokens like [CLS], [SEP], etc. Special Tokens are
                defined in self.all_special_tokens
        Returns:
            split_tokens (:obj:`list` of `str`): List of parsed tokens
        """
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                text,
                never_split=set(self.all_special_tokens + list(self.speaker2id.keys()) if not self.old_behaviour else [SPEAKER_TOKENS.ENTITY_1, SPEAKER_TOKENS.ENTITY_2])
                if not split_special_tokens
                else None,
            ):  
                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def is_speaker(self, token):
        r"""Check if the token matches a pre-defined speaker in speaker2id"""
        # temporarily only matters for checking entity_1 and entity_2
        return token in self.speaker2id

    def convert_speaker_to_id(self, token):
        r"""Convert token to a pre-defined speaker using speaker2id"""
        # entity 1 and entity 2 are unused2 and unused3 in original code, using speaker ids 11 and 12. returns token as is if not speaker
        if (not self.old_behaviour) or (token==SPEAKER_TOKENS.ENTITY_1 or token==SPEAKER_TOKENS.ENTITY_2):
            return self.speaker2id[token]
        else:
            return token

    def _convert_token_to_id(self, token):
        r"""Converts a token (str) in an id using the vocab.

        If the token maps to a valid speaker, it is converted to its respective speaker id instead.
        """
        if (self.old_behaviour and (token==SPEAKER_TOKENS.ENTITY_1 or token==SPEAKER_TOKENS.ENTITY_2)):
            # The original paper uses ids 2 and 3 as entity tokens in input_ids
            return 2 if token==SPEAKER_TOKENS.ENTITY_1 else 3
        if (not self.old_behaviour and token in self.speaker2id):
            return self.speaker2id[token]
        else:
            return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        r"""Converts an index (integer) in a token (str) using the vocab.

        If the id maps to a speaker, it is converted to its respective speaker token instead.
        """
        if (self.old_behaviour and (index==2 or index==3)):
            return SPEAKER_TOKENS.ENTITY_1 if index==2 else SPEAKER_TOKENS.ENTITY_2
        elif (not self.old_behaviour and str(index) in self.id2speaker):
            return self.speaker2id[str(index)]
        else:
            return self.ids_to_tokens.get(index, self.unk_token)