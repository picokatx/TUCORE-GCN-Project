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
    BertEmbeddings
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
        use_flash_attn=True,
        fused_bias_fc = True,
        fused_mlp = True,
        fused_dropout_add_ln = True,
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
        self.use_flash_attn=use_flash_attn,
        self.fused_bias_fc = fused_bias_fc,
        self.fused_mlp = fused_mlp,
        self.fused_dropout_add_ln = fused_dropout_add_ln,
        self.id2token = id2token
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict


def create_mixer_cls(config, cross_attn=False, return_residual=False):
    use_flash_attn = getattr(config, "use_flash_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    rotary_kwargs = {}
    if config.position_embedding_type == "rotary":
        rotary_kwargs["rotary_emb_dim"] = getattr(config, "rotary_emb_dim", config.hidden_size)
        rotary_kwargs["rotary_emb_base"] = getattr(config, "rotary_emb_base", 10000.0)
        rotary_kwargs["rotary_emb_scale_base"] = getattr(config, "rotary_emb_scale_base", None)
        rotary_kwargs["rotary_emb_interleaved"] = getattr(config, "rotary_emb_interleaved", False)
    mixer_cls = partial(
        MHA,
        num_heads=config.num_attention_heads,
        cross_attn=cross_attn,
        dropout=config.attention_probs_dropout_prob,
        causal=False,
        fused_bias_fc=fused_bias_fc,
        use_flash_attn=use_flash_attn,
        return_residual=return_residual,
        **rotary_kwargs,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, return_residual=False):
    inner_dim = config.intermediate_size
    fused_mlp = getattr(config, "fused_mlp", False)
    if fused_mlp:
        assert config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"], (
            "fused_mlp only " "supports approximate gelu"
        )
    if not fused_mlp:
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate=approximate),
            return_residual=return_residual,
        )
    else:
        if FusedMLP is None:
            raise ImportError("fused_dense is not installed")
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        mlp_cls = partial(
            FusedMLP,
            hidden_features=inner_dim,
            checkpoint_lvl=mlp_checkpoint_lvl,
            return_residual=return_residual,
        )
    return mlp_cls


def create_block(config, layer_idx=None):
    last_layer_subset = getattr(config, "last_layer_subset", False)
    cross_attn = last_layer_subset and layer_idx == config.num_hidden_layers - 1
    # TD [2022-12-19]: For cross attention (last layer), we actually want to return the
    # residual x_kv, not residual x. But it's annoying to change the API (and it only affects
    # one layer) so we just choose not to return residual in this case.
    return_residual = not cross_attn
    mixer_cls = create_mixer_cls(config, cross_attn, return_residual=return_residual)
    mlp_cls = create_mlp_cls(config, layer_idx, return_residual=return_residual)
    norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_eps)
    block = Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=False,
        resid_dropout1=config.hidden_dropout_prob,
        resid_dropout2=config.hidden_dropout_prob,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        return_residual=return_residual,
    )
    return block


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])

# class BertEmbeddings(nn.Module):
#     """Construct the embeddings from word, position and token_type embeddings."""

#     def __init__(self, config):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(
#             config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
#         )
#         self.position_embeddings = nn.Embedding(
#             config.max_position_embeddings, config.hidden_size
#         )
#         self.token_type_embeddings = nn.Embedding(
#             config.type_vocab_size, config.hidden_size
#         )
#         self.speaker_embeddings = nn.Embedding(
#             config.max_position_embeddings, config.hidden_size
#         )
#         # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
#         # any TensorFlow checkpoint file
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         # position_ids (1, len position emb) is contiguous in memory and exported when serialized
#         self.position_embedding_type = getattr(
#             config, "position_embedding_type", "absolute"
#         )
#         self.register_buffer(
#             "position_ids",
#             torch.arange(config.max_position_embeddings).expand((1, -1)),
#             persistent=False,
#         )
#         self.register_buffer(
#             "token_type_ids",
#             torch.zeros(self.position_ids.size(), dtype=torch.long),
#             persistent=False,
#         )

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         speaker_ids: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         past_key_values_length: int = 0,
#     ) -> torch.Tensor:
#         if input_ids is not None:
#             input_shape = input_ids.size()
#         else:
#             input_shape = inputs_embeds.size()[:-1]

#         seq_length = input_shape[1]

#         if position_ids is None:
#             position_ids = self.position_ids[
#                 :, past_key_values_length : seq_length + past_key_values_length
#             ]

#         # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
#         # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
#         # issue #5664
#         if token_type_ids is None:
#             if hasattr(self, "token_type_ids"):
#                 buffered_token_type_ids = self.token_type_ids[:, :seq_length]
#                 buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
#                     input_shape[0], seq_length
#                 )
#                 token_type_ids = buffered_token_type_ids_expanded
#             else:
#                 token_type_ids = torch.zeros(
#                     input_shape, dtype=torch.long, device=self.position_ids.device
#                 )
#         if inputs_embeds is None:
#             inputs_embeds = self.word_embeddings(input_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)
#         embeddings = inputs_embeds + token_type_embeddings
#         if self.position_embedding_type == "absolute":
#             position_embeddings = self.position_embeddings(position_ids)
#             embeddings += position_embeddings
#         speaker_embeddings = self.speaker_embeddings(speaker_ids)
#         embeddings += speaker_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.use_flash_attn = getattr(config, "use_flash_attn", False)
        self.layers = nn.ModuleList(
            [create_block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, key_padding_mask=None, subset_mask=None):
        """If subset_mask is not None, we only want output for the subset of the sequence.
        This means that we only compute the last layer output for these tokens.
        subset_mask: (batch, seqlen), dtype=torch.bool
        """
        if key_padding_mask is None or not self.use_flash_attn:
            mixer_kwargs = (
                {"key_padding_mask": key_padding_mask} if key_padding_mask is not None else None
            )
            for layer in self.layers:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
            if subset_mask is not None:
                hidden_states = hidden_states[subset_mask]
        else:
            batch, seqlen = hidden_states.shape[:2]
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                hidden_states, key_padding_mask
            )
            mixer_kwargs = {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen_in_batch}
            if subset_mask is None:
                for layer in self.layers:
                    hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                hidden_states = pad_input(hidden_states, indices, batch, seqlen)
            else:
                for layer in self.layers[:-1]:
                    hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                if key_padding_mask is not None:
                    subset_idx = torch.nonzero(
                        subset_mask[key_padding_mask], as_tuple=False
                    ).flatten()
                    subset_seqlens = (subset_mask & key_padding_mask).sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32), (1, 0)
                    )
                else:
                    subset_idx = torch.nonzero(subset_mask, as_tuple=False).flatten()
                    subset_seqlens = subset_mask.sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32), (1, 0)
                    )
                hidden_states_subset, hidden_states = index_first_axis_residual(
                    hidden_states, subset_idx
                )
                # It's ok to set max_seqlen_q to be much larger
                mixer_kwargs = {
                    "x_kv": hidden_states,
                    "cu_seqlens": subset_cu_seqlens,
                    "max_seqlen": max_seqlen_in_batch,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_k": max_seqlen_in_batch,
                }
                hidden_states = self.layers[-1](hidden_states_subset, mixer_kwargs=mixer_kwargs)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        self.transform_act_fn = nn.GELU(approximate=approximate)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        if not self.fused_dropout_add_ln:
            hidden_states = self.layer_norm(hidden_states)
        else:
            hidden_states = layer_norm(
                hidden_states, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps
            )
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense

        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = linear_cls(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPretraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        load_return = model.load_state_dict(
            remap_state_dict(state_dict_from_pretrained(model_name), config), strict=False
        )
        logger.info(load_return)
        return model
    

class BertModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig, add_pooling_layer=True):
        super().__init__(config)
        self.pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        if config.vocab_size % self.pad_vocab_size_multiple != 0:
            config.vocab_size += self.pad_vocab_size_multiple - (
                config.vocab_size % self.pad_vocab_size_multiple
            )
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        assert config.hidden_act in ["gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"]

        self.embeddings = BertEmbeddings(config)
        self.emb_drop = nn.Dropout(config.hidden_dropout_prob)
        self.emb_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.apply(partial(_init_weights, initializer_range=config.initializer_range))

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        masked_tokens_mask=None,
    ):
        """If masked_tokens_mask is not None (i.e. last_layer_subset == True in BertForPreTraining),
        we only want the output for the masked tokens. This means that we only compute the last
        layer output for these tokens.
        masked_tokens_mask: (batch, seqlen), dtype=torch.bool
        """
        hidden_states = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        # TD [2022-12:18]: Don't need to force residual in fp32
        # BERT puts embedding LayerNorm before embedding dropout.
        if not self.fused_dropout_add_ln:
            hidden_states = self.emb_ln(hidden_states)
        else:
            hidden_states = layer_norm(
                hidden_states, self.emb_ln.weight, self.emb_ln.bias, self.emb_ln.eps
            )
        hidden_states = self.emb_drop(hidden_states)

        if masked_tokens_mask is not None:
            batch_size, seqlen = input_ids.shape[:2]
            # We also need the first column for the CLS token
            first_col_mask = torch.zeros(
                batch_size, seqlen, dtype=torch.bool, device=input_ids.device
            )
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask
        else:
            subset_mask = None

        sequence_output = self.encoder(
            hidden_states, key_padding_mask=attention_mask, subset_mask=subset_mask
        )

        if masked_tokens_mask is None:
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            if attention_mask is not None:
                subset_idx = subset_mask[attention_mask]
                pool_input = sequence_output[first_col_mask[attention_mask][subset_idx]]
                sequence_output = sequence_output[masked_tokens_mask[attention_mask][subset_idx]]
            else:
                pool_input = sequence_output[first_col_mask[subset_mask]]
                sequence_output = sequence_output[masked_tokens_mask[subset_mask]]
            pooled_output = self.pooler(pool_input, pool=False) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )

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


import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat

from flash_attn.utils.distributed import get_dim_for_local_rank

try:
    from flash_attn import (
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )
except ImportError:
    flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func = None, None
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None
    flash_attn_with_kvcache = None

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear, FusedDense, RowParallelLinear
except ImportError:
    FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

# From https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )


class FlashSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    ):
        super().__init__()
        assert flash_attn_varlen_qkvpacked_func is not None, "FlashAttention is not installed"
        assert flash_attn_qkvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )
        else:
            return flash_attn_qkvpacked_func(
                qkv,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )

class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
            )
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output

class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input

def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        kv_cache = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        if use_alibi:
            assert use_flash_attn, "ALiBi code path requires flash_attn"
            alibi_slopes = torch.tensor(get_alibi_slopes(num_heads), device=device)
        else:
            alibi_slopes = None
        if window_size != (-1, -1):
            assert use_flash_attn, "Local (sliding window) attention code path requires flash_attn"

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert not cross_attn, "MHA with rotary embedding does not support cross-attention yet"
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = (
            LinearResidual if not fused_bias_fc else partial(FusedDense, return_residual=True)
        )
        wqkv_cls = linear_cls if not self.return_residual else linear_resid_cls
        inner_attn_cls = (
            partial(FlashSelfAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else SelfAttention
        )
        if not self.cross_attn:
            self.Wqkv = wqkv_cls(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        else:
            self.Wq = linear_cls(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
            self.Wkv = wqkv_cls(embed_dim, kv_dim, bias=qkv_proj_bias, **factory_kwargs)
        if self.dwconv:
            if self.num_heads_kv == self.num_heads:
                self.dwconv_qkv = nn.Conv1d(
                    qkv_dim, qkv_dim, kernel_size=3, padding=2, groups=qkv_dim
                )
            else:
                self.dwconv_q = nn.Conv1d(
                    embed_dim, embed_dim, kernel_size=3, padding=2, groups=embed_dim
                )
                self.dwconv_kv = nn.Conv1d(kv_dim, kv_dim, kernel_size=3, padding=2, groups=kv_dim)
        self.inner_attn = inner_attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
        )
        self.out_proj = linear_cls(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert not self.dwconv, "Generation does not support dwconv yet"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        assert self.use_flash_attn
        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, "This code path does not support xPos"
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.inner_cross_attn.softmax_scale,
            causal=self.inner_cross_attn.causal,
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
            alibi_slopes=alibi_slopes,
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
            or not self.use_flash_attn
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=self.inner_cross_attn.causal,
                alibi_slopes=alibi_slopes,
            )

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert not self.dwconv
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None
            assert not self.dwconv

        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask, **kwargs}
        )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        batch, seqlen = x.shape[:2]
        if not self.cross_attn and self.num_heads_kv == self.num_heads:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            if self.dwconv:
                qkv = rearrange(
                    self.dwconv_qkv(rearrange(qkv, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
            qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(
                        qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    context = self._update_kvcache_attention(
                        qkv[:, :, 0], qkv[:, :, 1:], inference_params
                    )
            else:
                context = self._apply_rotary_update_kvcache_attention(
                    qkv[:, :, 0], qkv[:, :, 1:], inference_params
                )
        else:
            if self.cross_attn:
                if not self.return_residual:
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
                    kv = self.Wkv(x_kv if x_kv is not None else x)
                else:
                    if x_kv is not None:
                        kv, x_kv = self.Wkv(x_kv)
                    else:
                        kv, x = self.Wkv(x)
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
            else:
                assert self.num_heads_kv != self.num_heads
                if not self.return_residual:
                    qkv = self.Wqkv(x)
                else:
                    qkv, x = self.Wqkv(x)
                q = qkv[..., : self.num_heads * self.head_dim]
                kv = qkv[..., self.num_heads * self.head_dim :]
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
            if self.dwconv:
                q = rearrange(
                    self.dwconv_q(rearrange(q, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
                kv = rearrange(
                    self.dwconv_kv(rearrange(kv, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(
                        q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(
                            self.inner_cross_attn, q, kv, **kwargs
                        )
                else:
                    context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)

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

class BertSpeakerEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.speaker_embeddings = nn.Embedding()
    def forward(
        self,
        speaker_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        
        return self.speaker_embeddings(speaker_ids)

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
        self.turnAttention = create_mixer_cls(config, False, return_residual=False)(config.hidden_size)

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
        embeddings = self.speaker_embeddings(speaker_ids)
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
