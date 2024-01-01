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

import numpy as np

from transformers import Pipeline
from tucore_gcn_transformers.tucore_gcn_bert_tokenizer import (
    SPEAKER_TOKENS,
    SpeakerBertTokenizer,
)
from tucore_gcn_transformers.tucore_gcn_bert_processor import (
    Conversation,
    SpeakerRelation,
    Message,
)
from collections import defaultdict
from typing import Any, Dict, List, Union
from itertools import permutations
import dgl
import torch.nn.functional as F
from torch import LongTensor


def create_inputs(conversation: Conversation, speaker_tokenizer: SpeakerBertTokenizer):
    inputs = conversation.build_inputs(speaker_tokenizer)[0]
    sequence = speaker_tokenizer.tokenize(inputs["dialog"])
    entity_1 = speaker_tokenizer.tokenize(inputs["relation"].entity_1)
    entity_2 = speaker_tokenizer.tokenize(inputs["relation"].entity_2)
    return inputs, sequence, entity_1, entity_2


def create_tokens(sequence, entity_1, entity_2):
    return (
        ["[CLS]"] + sequence + ["[SEP]"] + entity_1 + ["[SEP]"] + entity_2 + ["[SEP]"]
    )


def create_input_ids(tokens, speaker_tokenizer) -> List[int]:
    return speaker_tokenizer.convert_tokens_to_ids(tokens)


def create_input_mask(sequence, entity_1, entity_2):
    return (
        [1]
        + [1] * len(sequence)
        + [1]
        + [1] * len(entity_1)
        + [1]
        + [1] * len(entity_2)
        + [0]
    )


def create_segment_ids(sequence, entity_1, entity_2):
    return (
        [0]
        + [0] * len(sequence)
        + [0]
        + [1] * len(entity_1)
        + [1]
        + [1] * len(entity_2)
        + [1]
    )


def create_speaker_ids(
    sequence,
    entity_1,
    entity_2,
    entity_1_raw,
    entity_2_raw,
    speaker_tokenizer,
    old_behaviour,
):
    input_speaker_ids = []
    current_speaker_id = 0
    for token in sequence:
        if speaker_tokenizer.is_speaker(token):
            current_speaker_id = speaker_tokenizer.convert_speaker_to_id(token)
        input_speaker_ids.append(current_speaker_id)
    if old_behaviour:
        return (
            [0]
            + input_speaker_ids
            + [0]
            + [
                speaker_tokenizer.convert_speaker_to_id(entity_1_raw)
                if speaker_tokenizer.is_speaker(entity_1_raw)
                else 0
            ]
            * len(entity_1)
            + [0]
            + [
                speaker_tokenizer.convert_speaker_to_id(entity_2_raw)
                if speaker_tokenizer.is_speaker(entity_2_raw)
                else 0
            ]
            * len(entity_2)
            + [0]
        )
    else:
        return (
            [0]
            + input_speaker_ids
            + [0]
            + [0] * len(entity_1)
            + [0]
            + [0] * len(entity_2)
            + [0]
        )


def create_mention_ids(sequence, entity_1, entity_2, speaker_tokenizer, old_behaviour):
    input_mention_ids = []
    current_speaker_idx = 0
    for token in sequence:
        if speaker_tokenizer.is_speaker(token):
            current_speaker_idx += 1
        input_mention_ids.append(current_speaker_idx)
    if old_behaviour:
        return (
            [0]
            + input_mention_ids
            + [0]
            + [current_speaker_idx + 1] * len(entity_1)
            + [0]
            + [current_speaker_idx + 2] * len(entity_2)
            + [0]
        )
    else:
        return (
            [0]
            + input_mention_ids
            + [0]
            + [0] * len(entity_1)
            + [0]
            + [0] * len(entity_2)
            + [0]
        )


def create_label_id(rid, n_class):
    label_id = []
    for k in range(n_class):
        if k + 1 in rid:
            label_id.append(1)
        else:
            label_id.append(0)
    return label_id


def create_turn_mask(mention_id, old_behaviour=False):
    r"""Build speaker-dialog edge

    Adapted from [https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py]. Previously named mention2mask

    Creates a mapping of dialog indexes (mention_id) to each speaker_id.

    Usage:
    tokens: [CLS] {speaker_1} Howdy ! I ' m Flowey , Flowey the Flower ! {speaker_2} Hello Flowey ! [SEP] speaker_2 [SEP] Flowey [SEP] [PAD] ... [PAD]

    ```python
    >>> speaker_id = [0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,2,0,0,0,0,(...),0]
    >>> mention_id = [0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,3,0,0,0,0,(...),0]
    >>> make_speaker_infor(speaker_id, mention_id)
    >>> {1: [1], 2: [2]}
    ```
    """
    slen = len(mention_id)
    mask = []
    if old_behaviour:
        turn_mention_ids = [i for i in range(1, np.max(mention_id) - 1)]  # -1
    else:
        turn_mention_ids = [i for i in range(1, np.max(mention_id) + 1)]  # -1
    for j in range(slen):
        tmp = None
        if mention_id[j] not in turn_mention_ids:
            tmp = np.zeros(slen, dtype=bool)
            if old_behaviour:
                tmp[j] = 1
        else:
            start = mention_id[j]
            end = mention_id[j]
            if mention_id[j] - 1 in turn_mention_ids:
                start = mention_id[j] - 1
            if mention_id[j] + 1 in turn_mention_ids:
                end = mention_id[j] + 1
            tmp = (mention_id >= start) & (mention_id <= end)
        mask.append(tmp)
    mask = np.stack(mask)
    return mask


def pad_inputs(
    tokens: List[int],
    input_ids,
    input_mask,
    segment_ids,
    speaker_ids,
    mention_ids,
    max_seq_length,
):
    while len(input_ids) < max_seq_length:
        tokens.append("[PAD]")
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)


def build_speaker_turn_relations(speaker_id: List[int], mention_id: List[int]):
    r"""Build speaker-dialog edge data

    Adapted from from [https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py] Previously named make_speaker_infor

    Creates a mapping of turn indexes (mention_id) to each speaker_id.

    Modifications Summary:
        No changes were made to this function.

    Usage:
    tokens: [CLS] {speaker_1} Howdy ! I ' m Flowey , Flowey the Flower ! {speaker_2} Hello Flowey ! [SEP] speaker_2 [SEP] Flowey [SEP] [PAD] ... [PAD]

    ```python
    >>> speaker_id = [0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,2,0,0,0,0,(...),0]
    >>> mention_id = [0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,3,0,4,0,0,(...),0]
    >>> build_speaker_turn_relations(speaker_id, mention_id)
    {1: [1], 2: [2]}
    ```
    """
    tmp = defaultdict(set)
    for i in range(1, len(speaker_id)):
        if (
            speaker_id[i] == 0
        ):  # End of actual input, we ignore [SEP]S_1[SEP]S_2[SEP] portion
            break
        tmp[speaker_id[i]].add(mention_id[i])
    speaker_turn_edges = dict()
    for k, va in tmp.items():
        speaker_turn_edges[k] = list(va)
    return speaker_turn_edges


def build_entity_turn_relations(
    input_ids: List[int],
    mention_ids: List[int],
    entity_1: List[int],
    entity_2: List[int],
):
    r"""Build speaker-dialog edge data

    Adapted from from [https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py]. Previously named make_entity_edges_infor

    Creates a mapping of dialog indexes (mention_id) to each speaker_id.

    Modifications Summary:
        added entity_1 and entity_2 args so we do not have to parse mention_ids for them.
        renamed return dict to entity_1, entity_2 for consistency

    Usage:
    tokens: [CLS] {speaker_1} howdy ! i ' m flowey , flowey the flower ! {speaker_2} hello flowey ! [SEP] {speaker_2} [SEP] flowey [SEP] [PAD] (...) [PAD]

    ```python
    >>> input_ids = [101, 1, 100, 999, 1045, 1005, 1049, 100, 1010, 100, 1996, 6546, 999, 2, 7592, 100, 999, 102, 2, 102, 100, 102, 0, (...), 0]
    >>> speaker_id = [0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,2,0,0,0,0,(...),0]
    >>> mention_id = [0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,3,0,4,0,0,(...),0]
    >>> build_entity_turn_relations(input_ids, mention_id, [2], [100])
    >>> {'h': [2, 3], 't': [1, 1, 1, 2, 4]}
    ```
    """
    entity_turn_edges = {"entity_1": [], "entity_2": []}
    for i in range(len(input_ids) - len(entity_1)):
        if input_ids[i : i + len(entity_1)] == entity_1:
            entity_turn_edges["entity_1"].append(mention_ids[i])
    for i in range(len(input_ids) - len(entity_2)):
        if input_ids[i : i + len(entity_2)] == entity_2:
            entity_turn_edges["entity_2"].append(mention_ids[i])
    return entity_turn_edges


def create_speaker_turn_edges(graph_data, speaker_turn_relations):
    for _, mentions in speaker_turn_relations.items():
        for entity_1, entity_2 in permutations(mentions, 2):
            graph_data[("node", "speaker", "node")].append((entity_1, entity_2))
    # If there's no speaker-turn relations to add, we connect turn 1 to turn 0 ([CLS] token, used as document node, possibly assumes that there is only 1 speaker and 1 turn)
    if graph_data[("node", "speaker", "node")] == []:
        graph_data[("node", "speaker", "node")].append((1, 0))
    return graph_data


def create_document_turn_edges(graph_data, turn_node_num):
    # Connect all turns to turn 0 ([CLS] token, used as document node)
    for i in range(1, turn_node_num + 1):
        graph_data[("node", "dialog", "node")].append((i, 0))
        graph_data[("node", "dialog", "node")].append((0, i))
    if graph_data[("node", "dialog", "node")] == []:
        graph_data[("node", "dialog", "node")].append((1, 0))
    return graph_data


def create_entity_turn_edges(
    graph_data, entity_turn_relations, entity_1_mention_id, entity_2_mention_id
):
    # Connect all turn nodes owned by single speakers to each other.
    for mention in entity_turn_relations["entity_1"]:
        graph_data[("node", "entity", "node")].append((entity_1_mention_id, mention))
        graph_data[("node", "entity", "node")].append((mention, entity_1_mention_id))
    for mention in entity_turn_relations["entity_2"]:
        graph_data[("node", "entity", "node")].append((entity_2_mention_id, mention))
        graph_data[("node", "entity", "node")].append((mention, entity_2_mention_id))
    if entity_turn_relations["entity_1"] == []:
        graph_data[("node", "entity", "node")].append((entity_1_mention_id, 0))
        graph_data[("node", "entity", "node")].append((0, entity_1_mention_id))
    if entity_turn_relations["entity_2"] == []:
        graph_data[("node", "entity", "node")].append((entity_2_mention_id, 0))
        graph_data[("node", "entity", "node")].append((0, entity_2_mention_id))
    return graph_data


def create_graph(
    input_ids,
    speaker_ids,
    mention_ids,
    entity_1_ids,
    entity_2_ids,
    turn_node_num,
    entity_1_mention_id,
    entity_2_mention_id,
):
    r"""Build Dialog Graph

    Adapted from from [https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py]

    Creates Dialogue graph as specified by TUCORE-GCN

    Usage:

    ```python
    >>> graph = create_graph(
            input_ids,
            speaker_ids,
            mention_ids,
            entity_1_ids,
            entity_2_ids,
            turn_node_num,
            entity_1_mention_id,
            entity_2_mention_id,
        )
    Graph(num_nodes={'node': 5},
    num_edges={('node', 'dialog', 'node'): 4, ('node', 'entity', 'node'): 8, ('node', 'speaker', 'node'): 1},
    metagraph=[('node', 'node', 'dialog'), ('node', 'node', 'entity'), ('node', 'node', 'speaker')])
    ```
    """
    graph_data = defaultdict(list)
    speaker_turn_relations = build_speaker_turn_relations(speaker_ids, mention_ids)
    entity_turn_relations = build_entity_turn_relations(
        input_ids, mention_ids, entity_1_ids, entity_2_ids
    )
    graph_data = create_speaker_turn_edges(graph_data, speaker_turn_relations)
    graph_data = create_document_turn_edges(graph_data, turn_node_num)
    graph_data = create_entity_turn_edges(
        graph_data, entity_turn_relations, entity_1_mention_id, entity_2_mention_id
    )
    graph = dgl.heterograph(graph_data)
    return graph


class ConversationalSequenceClassificationPipeline(Pipeline):
    r"""transformers pipeline for TUCORE-GCN. Generalizable to other models in theory

    Adapted from [https://github.com/BlackNoodle/TUCORE-GCN/blob/main/data.py]

    Defines preprocessing, model execution and postprocessing code for transformers Pipelines.

    Usage:

    ```python
    >>> PIPELINE_REGISTRY.register_pipeline(
        "conversational-sequence-classification",
        pipeline_class=ConversationalSequenceClassificationPipeline
        )
    >>> speaker_tokenizer = SpeakerBertTokenizer.from_pretrained('bert-base-uncased')
    >>> model = TUCOREGCN_BertForSequenceClassification(TUCOREGCN_BertConfig.from_json_file("../models/BERT/tucoregcn_bert_mlc.json"))
    >>> model.cuda()
    >>> model.load_state_dict(torch.load("../TUCOREGCN_BERT_DialogRE/tucoregcn_pytorch_model.pt"))
    >>> model.cuda()
    >>> classifier = pipeline("conversational-sequence-classification", model=model, tokenizer=speaker_tokenizer, device="cuda:0", n_class=36, max_seq_length=512)
    >>> c = Conversation(
            messages=[
                Message("Speaker 1", "Howdy! I'm Flowey, Flowey the Flower!"),
                Message("Speaker 2", "Hello Flowey. I'm your very best friend!"),
            ],
            speaker_relations=[
                SpeakerRelation("Speaker 1", "Speaker 2")
            ]
        )
    >>> labels, scores, logits = classifier(c).values()
    >>> print(labels, scores)
    LABEL_17 34.35903269029420
    ```
    """
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "n_class" in kwargs:
            preprocess_kwargs["n_class"] = kwargs["n_class"]
        if "max_seq_length" in kwargs:
            preprocess_kwargs["max_seq_length"] = kwargs["max_seq_length"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, conversation: Conversation, n_class:int, max_seq_length:int):
        speaker_tokenizer = self.tokenizer
        old_behaviour = True
        inputs, sequence, entity_1, entity_2 = create_inputs(
            conversation, speaker_tokenizer
        )
        tokens = create_tokens(sequence, entity_1, entity_2)
        input_ids = create_input_ids(tokens, speaker_tokenizer)
        input_mask = create_input_mask(sequence, entity_1, entity_2)
        segment_ids = create_segment_ids(sequence, entity_1, entity_2)
        speaker_ids = create_speaker_ids(
            sequence,
            entity_1,
            entity_2,
            inputs["relation"].entity_1,
            inputs["relation"].entity_2,
            speaker_tokenizer,
            old_behaviour,
        )
        mention_ids = create_mention_ids(
            sequence, entity_1, entity_2, speaker_tokenizer, old_behaviour
        )
        label_id = create_label_id(inputs["relation"].rid, n_class)
        print(            tokens,
            input_ids,
            input_mask,
            segment_ids,
            speaker_ids,
            mention_ids,
            max_seq_length,)
        pad_inputs(
            tokens,
            input_ids,
            input_mask,
            segment_ids,
            speaker_ids,
            mention_ids,
            max_seq_length,
        )
        turn_masks = create_turn_mask(np.array(mention_ids), old_behaviour)

        turn_node_num = max(mention_ids) - 2
        entity_1_mention_id = max(mention_ids) - 1
        entity_2_mention_id = max(mention_ids)
        entity_1_ids = speaker_tokenizer.convert_tokens_to_ids(entity_1)
        entity_2_ids = speaker_tokenizer.convert_tokens_to_ids(entity_2)
        graph = create_graph(
            input_ids,
            speaker_ids,
            mention_ids,
            entity_1_ids,
            entity_2_ids,
            turn_node_num,
            entity_1_mention_id,
            entity_2_mention_id,
        )
        # Checks that the number of nodes referenced is turn_node_num+1(entity_1)+1(entity_2)+1(document_node)
        # The reason this works is that max(mention_ids) returns the id of entity 2. hence,
        # assert len(used_mention) == (max(mention_ids) + 1)

        return (
            [tokens],
            np.array([input_ids]),
            np.array([input_mask]),
            np.array([segment_ids]),
            np.array([speaker_ids]),
            np.array([mention_ids]),
            np.array([turn_masks]),
            [graph],
        )

    def _forward(self, model_inputs):  # model_inputs == {"model_input": model_input}
        (
            tokens,
            input_ids,
            input_mask,
            segment_ids,
            speaker_ids,
            mention_ids,
            turn_masks,
            graph,
        ) = model_inputs
        output = self.model(
            LongTensor(input_ids).to(self.device),
            LongTensor(segment_ids).to(self.device),
            LongTensor(input_mask).to(self.device),
            LongTensor(speaker_ids).to(self.device),
            [g.to(self.device) for g in graph],
            LongTensor(mention_ids).to(self.device),
            None,
            LongTensor(turn_masks).to(self.device),
        )
        output
        return output

    def postprocess(self, model_outputs):
        logits = model_outputs.logits
        probabilities = F.softmax(logits, dim=1)
        best_class = probabilities.argmax(1)
        label = [
            self.model.config.id2label[int(cid.cpu().numpy())] for cid in best_class
        ]
        score = [
            probabilities[i][best_class[i]].item() for i in range(len(probabilities))
        ]
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
