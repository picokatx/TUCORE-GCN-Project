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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers.models.llama.tokenization_llama import LlamaTokenizer, SPIECE_UNDERLINE
from transformers.tokenization_utils import AddedToken
import regex as re
if TYPE_CHECKING:
    from transformers.tokenization_utils_base import TextInput

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
# <s>{speaker}: Howdy {speaker}!</s>{speaker}: Hello {speaker}!</s>0</s>2</s>

class SpeakerLlamaTokenizer(LlamaTokenizer):
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
        SPEAKER_TOKENS.SPEAKER_1: 21,
        SPEAKER_TOKENS.SPEAKER_2: 22,
        SPEAKER_TOKENS.SPEAKER_3: 23,
        SPEAKER_TOKENS.SPEAKER_4: 24,
        SPEAKER_TOKENS.SPEAKER_5: 25,
        SPEAKER_TOKENS.SPEAKER_6: 26,
        SPEAKER_TOKENS.SPEAKER_7: 27,
        SPEAKER_TOKENS.SPEAKER_8: 28,
        SPEAKER_TOKENS.SPEAKER_9: 29,
    }

    id2speaker = {
        "11": SPEAKER_TOKENS.ENTITY_1,
        "12": SPEAKER_TOKENS.ENTITY_2,
        "21": SPEAKER_TOKENS.SPEAKER_1,
        "22": SPEAKER_TOKENS.SPEAKER_2,
        "23": SPEAKER_TOKENS.SPEAKER_3,
        "24": SPEAKER_TOKENS.SPEAKER_4,
        "25": SPEAKER_TOKENS.SPEAKER_5,
        "26": SPEAKER_TOKENS.SPEAKER_6,
        "27": SPEAKER_TOKENS.SPEAKER_7,
        "28": SPEAKER_TOKENS.SPEAKER_8,
        "29": SPEAKER_TOKENS.SPEAKER_9,
    }
    def __init__(self, vocab_file, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token=None, sp_model_kwargs = None, add_bos_token=True, add_eos_token=False, clean_up_tokenization_spaces=True, use_default_system_prompt=False, spaces_between_special_tokens=False, legacy=None, add_prefix_space=True, **kwargs):
        super().__init__(vocab_file, unk_token, bos_token, eos_token, pad_token, sp_model_kwargs, add_bos_token, add_eos_token, clean_up_tokenization_spaces, use_default_system_prompt, spaces_between_special_tokens, legacy, add_prefix_space, **kwargs)
        self.model_max_length = 2048
        self.add_tokens({k: v for k,v in self.speaker2id.items()}, special_tokens=True)

    def add_tokens(self, new_tokens: Union[Dict[str, int], Dict[AddedToken, int]], special_tokens: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary. Special tokens are sometimes already in the
        vocab which is why they have to be handled specifically.

        Args:
            new_tokens (`List[str]`or `List[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is counted as added if it's not already in the vocabulary
                (tested by checking if the tokenizer assign the index of the `unk_token` to them). If a token is part
                of the vocabulary then we simply mark this token as an `AddedToken` which allows to control the
                stripping and normalization of this token. This is NOT possible in `tokenizers`.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the tokens should be added as special tokens.

        Returns:
            `int`: The number of tokens actually added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```
        """
        # TODO this is fairly slow to improve!
        current_vocab = self.get_vocab().copy()
        for token, token_index in new_tokens.items():
            if not isinstance(token, (str, AddedToken)):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if str(token) == "":
                continue
            if isinstance(token, str):
                if token in self._added_tokens_encoder:
                    continue
                else:
                    # very important for fast and slow equivalence!
                    is_special = token in self.all_special_tokens or special_tokens
                    token = AddedToken(
                        token, rstrip=False, lstrip=False, normalized=not is_special, special=is_special
                    )
            elif special_tokens:
                # doing token.special=True changes the normalization! will fix in rust
                # this is important and the only reason why the AddedTokens in each class are normalized by default
                token.__setstate__({"special": True, "normalized": token.normalized})
            if token in self._added_tokens_decoder:
                continue
            if not token.special and token.normalized and getattr(self, "do_lower_case", False):
                # Normalize if requested
                token.content = token.content.lower()
            if token.content not in current_vocab:
                current_vocab[token.content] = token_index
            else:
                token_index = current_vocab[token.content]

            if token.special and str(token) not in self.all_special_tokens:
                self._additional_special_tokens.append(token)
                self.all_special_tokens.append(token)
            # the setter automatically updates the reverse map
            self._added_tokens_decoder[token_index] = token
            self._added_tokens_encoder[token.content] = token_index

        self._update_trie()
        return self


    def is_speaker(self, token):
        r"""Check if the token matches a pre-defined speaker in speaker2id"""
        return token in self.speaker2id

	
    def convert_speaker_to_id(self, token):
        r"""Convert token to a pre-defined speaker using speaker2id"""
        # entity 1 and entity 2 are unused2 and unused3 in original code, using speaker ids 11 and 12. returns token as is if not speaker
        return self.speaker2id[token]
