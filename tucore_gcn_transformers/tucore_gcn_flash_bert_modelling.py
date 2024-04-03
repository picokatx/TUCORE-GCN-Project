# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BertConfig, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertForPreTrainingOutput,
    SequenceClassifierOutput,
    BertEmbeddings,
)

from flash_attn.bert_padding import (
    index_first_axis,
    index_first_axis_residual,
    pad_input,
    unpad_input,
)
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
from flash_attn.models.bert import BertModel
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers.models.bert.configuration_bert import BertConfig

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None
    
try:
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin")
    from flash_attn.ops.layer_norm import dropout_add_layer_norm, layer_norm
except ImportError:
    dropout_add_layer_norm, layer_norm = None, None

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None

import dgl
import dgl.nn.pytorch as dglnn

logger = logging.getLogger(__name__)

class TUCOREGCN_BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [bert-base-uncased](https://huggingface.co/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        gcn_layers (`int`, *optional*, defaults to 2),
        gcn_act (`str`, *optional*, defaults to `"relu"`),
        gcn_dropout (`int`, *optional*, defaults to 0.6),

    Examples:

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model (with random weights) from the bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        turn_attention_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        gcn_layers=2,
        gcn_act="relu",
        gcn_dropout=0.6,
        debug_ret=False,
        id2token={},
        output_attentions = True,
        output_hidden_states = True,
        return_dict = True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.turn_attention_dropout_prob = turn_attention_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.gcn_layers = gcn_layers
        self.gcn_act = gcn_act
        self.gcn_dropout = gcn_dropout
        self.debug_ret = debug_ret
        self.id2token = id2token
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict

class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        rel_names,
        num_bases,
        *,
        weight=True,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0,
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_feat, out_feat, norm="right", weight=False, bias=False
                )
                for rel in rel_names
            }
        )

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names)
                )
            else:
                self.weight = nn.Parameter(
                    torch.Tensor(len(self.rel_names), in_feat, out_feat)
                )
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain("relu")
                )

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        r"""Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {
                self.rel_names[i]: {"weight": w.squeeze(0)}
                for i, w in enumerate(torch.split(weight, 1, dim=0))
            }
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class TurnLevelLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, lstm_dropout, dropout_rate):
        super(TurnLevelLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm2hiddnesize = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, inputs):
        lstm_out = self.lstm(inputs)
        lstm_out = lstm_out[0].squeeze(0)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.bilstm2hiddnesize(lstm_out)
        return lstm_out

class TUCOREGCN_BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TUCOREGCN_BertConfig
    # load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.bias.data.zero_()

class TUCOREGCN_Bert(TUCOREGCN_BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.gcn_dim = config.hidden_size
        self.gcn_layers = config.gcn_layers
        
        self.speaker_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.use_cache = config.use_cache
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.return_dict

        if config.gcn_act == "tanh":
            self.activation = nn.Tanh()
        elif config.gcn_act == "relu":
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.turnAttention = MHA(
            config.num_attention_heads,
            config.hidden_size,
            self.attention_head_size,
            self.attention_head_size,
            config,
        )

        rel_name_lists = [
            "speaker",
            "dialog",
            "entity",
        ]  # entity: object/subject node as defined in paper
        self.GCN_layers = nn.ModuleList(
            [
                RelGraphConvLayer(
                    self.gcn_dim,
                    self.gcn_dim,
                    rel_name_lists,
                    num_bases=len(rel_name_lists),
                    activation=self.activation,
                    self_loop=True,
                    dropout=config.gcn_dropout,
                )
                for i in range(self.gcn_layers)
            ]
        )

        self.LSTM_layers = nn.ModuleList(
            [
                TurnLevelLSTM(config.hidden_size, 2, 0.2, 0.4)
                for i in range(self.gcn_layers)
            ]
        )
        # debug
        self.debug_ret = config.debug_ret

    r"""
    input_ids: denotes token embeddings
    token_type_ids: artifact from NSP subtask of BERT. In this model the first sentence is the full conversation, while the 2nd sentence is a person???
    attention_mask: an input mask is passed here
    speaker_ids: denotes the current turn's speaker's id
    graphs: for gcn
    mention_id: denotes the turn index
    turn_mask: denotes the current turn and the next turn as a token mask
    """

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        speaker_ids,
        graphs,
        mention_ids,
        turn_mask=None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
    ):
        embeddings = self.speaker_embeddings(speaker_ids)
        """
        Encoder Module
        """
        outputs = self.bert(
            input_ids,
            speaker_ids=speaker_ids,
            token_type_ids=token_type_ids,
            encoder_attention_mask=attention_mask,
            use_cache = self.use_cache,
            output_attentions = self.output_attentions,
            output_hidden_states = self.output_hidden_states,
            return_dict = self.return_dict,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            encoder_hidden_states = encoder_hidden_states,
            past_key_values = past_key_values,
        )
        sequence_outputs, pooled_outputs = (
            outputs.last_hidden_state,
            outputs.pooler_output,
        )
        """
        Turn Attention Module
        """
        sequence_outputs, attn = self.turnAttention(
            sequence_outputs, sequence_outputs, sequence_outputs, turn_mask
        )
        
        """
        Selectively obtain features from turn attention module output
        """
        # initialize some variables
        features = None
        num_batch_turn = []
        slen = input_ids.size(1)
        # Iterate over all inputs to be processed
        for i in range(len(graphs)):
            sequence_output = sequence_outputs[i]
            mention_id = mention_ids[i]
            pooled_output = pooled_outputs[i]
            # Find the last turn (in the case of tucoregcn, it is the masked speaker dictionary)
            mention_num = torch.max(mention_id)
            # Create a mention matrix idx of mention_num*slen
            num_batch_turn.append(mention_num + 1)
            mention_index = (
                (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen)
            )
            if torch.cuda.is_available():
                mention_index = mention_index.cuda()
            # Create a mentions matrix of slen*mention_num
            mentions = mention_id.unsqueeze(0).expand(mention_num, -1)
            # Generate truth matrix of each speaker's dialogue over the entire conversation
            select_metrix = (mention_index == mentions).float()
            # Factor into the one-hot encoding, the total number of words in each speaker's dialogue
            word_total_numbers = (
                torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)
            )
            select_metrix = torch.where(
                word_total_numbers > 0,
                select_metrix / word_total_numbers,
                select_metrix,
            )
            # Apply one hot encoding to sequence_output to selectively obtain features
            x = torch.mm(select_metrix, sequence_output)
            x = torch.cat((pooled_output.unsqueeze(0), x), dim=0)
            # Iteratively concatenate features from all sequence_outputs
            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)
        # Batch dgl graphs into 1 graph for efficient computation
        graph_big = dgl.batch(graphs)
        output_features = [features]
        """
        Dialogue Graph with Sequential Nodes Module
        """
        # Loop over 2 GCN Layers
        for layer_num, GCN_layer in enumerate(self.GCN_layers):
            start = 0
            new_features = []
            # Loop over inputs
            for idx in num_batch_turn:
                # CLS feature
                new_features.append(features[start])
                # Dialogue Feature
                lstm_out = self.LSTM_layers[layer_num](
                    features[start + 1 : start + idx - 2].unsqueeze(0)
                )
                new_features += lstm_out
                # Object+Subject Feature
                new_features.append(features[start + idx - 2])
                new_features.append(features[start + idx - 1])
                # Increment by length of current input feature
                start += idx
            # Throw R-GCN at it and pray
            features = torch.stack(new_features)
            features = GCN_layer(graph_big, {"node": features})["node"]
            output_features.append(features)
        """
        Put together return graph output
        """
        graphs = dgl.unbatch(graph_big)
        graph_output = list()
        fea_idx = 0
        for i in range(len(graphs)):
            node_num = graphs[i].number_of_nodes("node")
            intergrated_output = None
            for j in range(self.gcn_layers + 1):
                if intergrated_output == None:
                    intergrated_output = output_features[j][fea_idx]
                else:
                    intergrated_output = torch.cat(
                        (intergrated_output, output_features[j][fea_idx]), dim=-1
                    )
                intergrated_output = torch.cat(
                    (intergrated_output, output_features[j][fea_idx + node_num - 2]),
                    dim=-1,
                )
                intergrated_output = torch.cat(
                    (intergrated_output, output_features[j][fea_idx + node_num - 1]),
                    dim=-1,
                )
            fea_idx += node_num
            graph_output.append(intergrated_output)
        graph_output = torch.stack(graph_output)

        return BaseModelOutput(graph_output, outputs.past_key_values, attn)

# Partially copied from BertForSequenceClassification
class TUCOREGCNForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.tucoregcn_llm = TUCOREGCN_Bert(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(
            config.hidden_size * 3 * (config.gcn_layers + 1), config.num_labels
        )
        self.debug_ret = config.debug_ret
        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        graphs: Optional[torch.Tensor] = None,
        mention_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        turn_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
    
        outputs = self.tucoregcn_llm(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            speaker_ids=speaker_ids,
            graphs=graphs,
            mention_ids=mention_ids,
            turn_mask=turn_mask,
        )

        if self.debug_ret:
            return outputs

        pooled_output = outputs.last_hidden_state
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def remap_state_dict(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a Huggingface BERT model to be flash_attn compatible.
    """

    # LayerNorm
    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_ln_gamma_beta(k), v) for k, v in state_dict.items())

    # Layers
    def key_mapping_layers(key):
        return re.sub(r"^bert.encoder.layer.", "bert.encoder.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^bert.embeddings.LayerNorm.", "bert.emb_ln.", key)
        key = re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm2.\2",
            key,
        )
        key = re.sub(
            r"^cls.predictions.transform.LayerNorm.(weight|bias)",
            r"cls.predictions.transform.layer_norm.\1",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^bert.encoder.layers.(\d+).intermediate.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc2.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    last_layer_subset = getattr(config, "last_layer_subset", False)
    for d in range(config.num_hidden_layers):
        Wq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.weight")
        Wk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.weight")
        Wv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.weight")
        bq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.bias")
        bk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.bias")
        bv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.bias")
        if not (last_layer_subset and d == config.num_hidden_layers - 1):
            state_dict[f"bert.encoder.layers.{d}.mixer.Wqkv.weight"] = torch.cat(
                [Wq, Wk, Wv], dim=0
            )
            state_dict[f"bert.encoder.layers.{d}.mixer.Wqkv.bias"] = torch.cat([bq, bk, bv], dim=0)
        else:
            state_dict[f"bert.encoder.layers.{d}.mixer.Wq.weight"] = Wq
            state_dict[f"bert.encoder.layers.{d}.mixer.Wkv.weight"] = torch.cat([Wk, Wv], dim=0)
            state_dict[f"bert.encoder.layers.{d}.mixer.Wq.bias"] = bq
            state_dict[f"bert.encoder.layers.{d}.mixer.Wkv.bias"] = torch.cat([bk, bv], dim=0)

    def key_mapping_attn(key):
        return re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mixer.out_proj.\2",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_decoder_bias(key):
        return re.sub(r"^cls.predictions.bias", "cls.predictions.decoder.bias", key)

    state_dict = OrderedDict((key_mapping_decoder_bias(k), v) for k, v in state_dict.items())

    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["bert.embeddings.word_embeddings.weight"]
        state_dict["bert.embeddings.word_embeddings.weight"] = F.pad(
            word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0])
        )
        decoder_weight = state_dict["cls.predictions.decoder.weight"]
        state_dict["cls.predictions.decoder.weight"] = F.pad(
            decoder_weight, (0, 0, 0, config.vocab_size - decoder_weight.shape[0])
        )
        # If the vocab was padded, we want to set the decoder bias for those padded indices to be
        # strongly negative (i.e. the decoder shouldn't predict those indices).
        # TD [2022-05-09]: I don't think it affects the MLPerf training.
        decoder_bias = state_dict["cls.predictions.decoder.bias"]
        state_dict["cls.predictions.decoder.bias"] = F.pad(
            decoder_bias, (0, config.vocab_size - decoder_bias.shape[0]), value=-100.0
        )

    return state_dict


def inv_remap_state_dict(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a flash_attn model to be Huggingface BERT compatible.

    This function is meant to be the inverse of remap_state_dict.
    """
    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["bert.embeddings.word_embeddings.weight"]
        decoder_weight = state_dict["cls.predictions.decoder.weight"]
        decoder_bias = state_dict["cls.predictions.decoder.bias"]
        # unpad embeddings
        state_dict["bert.embeddings.word_embeddings.weight"] = word_embeddings[
            : config.orig_vocab_size, :
        ]
        state_dict["cls.predictions.decoder.weight"] = decoder_weight[: config.orig_vocab_size, :]
        state_dict["cls.predictions.decoder.bias"] = decoder_bias[: config.orig_vocab_size]

    for d in range(config.num_hidden_layers):
        last_layer_subset = getattr(config, "last_layer_subset", False)
        if not last_layer_subset or d != (config.num_hidden_layers - 1):
            Wqkv_weights = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wqkv.weight")
            Wqkv_biases = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wqkv.bias")
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.weight"] = Wqkv_weights[
                : Wqkv_weights.shape[0] // 3, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.weight"] = Wqkv_weights[
                Wqkv_weights.shape[0] // 3 : 2 * Wqkv_weights.shape[0] // 3, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.weight"] = Wqkv_weights[
                2 * Wqkv_weights.shape[0] // 3 :, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.bias"] = Wqkv_biases[
                : Wqkv_biases.shape[0] // 3
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.bias"] = Wqkv_biases[
                Wqkv_biases.shape[0] // 3 : 2 * Wqkv_biases.shape[0] // 3
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.bias"] = Wqkv_biases[
                2 * Wqkv_biases.shape[0] // 3 :
            ]
        else:
            Wq_weight = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wq.weight")
            Wkv_weights = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wkv.weight")
            Wq_bias = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wq.bias")
            Wkv_biases = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wkv.bias")
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.weight"] = Wq_weight
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.weight"] = Wkv_weights[
                : Wkv_weights.shape[0] // 2, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.weight"] = Wkv_weights[
                Wkv_weights.shape[0] // 2 :, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.bias"] = Wq_bias
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.bias"] = Wkv_biases[
                : Wkv_biases.shape[0] // 2
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.bias"] = Wkv_biases[
                Wkv_biases.shape[0] // 2 :
            ]

    def inv_key_mapping_ln(key):
        key = re.sub(r"bert.emb_ln.", "bert.embeddings.LayerNorm.", key)
        key = re.sub(
            r"bert.encoder.layers.(\d+).norm1.(weight|bias)",
            r"bert.encoder.layers.\1.attention.output.LayerNorm.\2",
            key,
        )
        key = re.sub(
            r"bert.encoder.layers.(\d+).norm2.(weight|bias)",
            r"bert.encoder.layers.\1.output.LayerNorm.\2",
            key,
        )
        key = re.sub(
            r"cls.predictions.transform.layer_norm.(weight|bias)",
            r"cls.predictions.transform.LayerNorm.\1",
            key,
        )
        return key

    def inv_key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.weight$", "LayerNorm.gamma", key)
        key = re.sub(r"LayerNorm.bias$", "LayerNorm.beta", key)
        return key

    def inv_key_mapping_layers(key):
        return re.sub(r"bert.encoder.layers.", "bert.encoder.layer.", key)

    def inv_key_mapping_mlp(key):
        key = re.sub(
            r"bert.encoder.layer.(\d+).mlp.fc1.(weight|bias)",
            r"bert.encoder.layer.\1.intermediate.dense.\2",
            key,
        )
        key = re.sub(
            r"bert.encoder.layer.(\d+).mlp.fc2.(weight|bias)",
            r"bert.encoder.layer.\1.output.dense.\2",
            key,
        )
        return key

    def inv_key_mapping_attn(key):
        return re.sub(
            r"bert.encoder.layer.(\d+).mixer.out_proj.(weight|bias)",
            r"bert.encoder.layer.\1.attention.output.dense.\2",
            key,
        )

    def inv_key_mapping_decoder_bias(key):
        return re.sub(r"cls.predictions.decoder.bias", "cls.predictions.bias", key)

    state_dict = OrderedDict((inv_key_mapping_ln(key), value) for key, value in state_dict.items())
    state_dict = OrderedDict(
        (inv_key_mapping_ln_gamma_beta(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict(
        (inv_key_mapping_layers(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict((inv_key_mapping_mlp(key), value) for key, value in state_dict.items())
    state_dict = OrderedDict(
        (inv_key_mapping_attn(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict(
        (inv_key_mapping_decoder_bias(key), value) for key, value in state_dict.items()
    )

    return state_dict
