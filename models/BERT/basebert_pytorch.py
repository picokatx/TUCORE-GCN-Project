# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
	BaseModelOutputWithPastAndCrossAttentions,
	BaseModelOutputWithPoolingAndCrossAttentions,
	CausalLMOutputWithCrossAttentions,
	MaskedLMOutput,
	MultipleChoiceModelOutput,
	NextSentencePredictorOutput,
	QuestionAnsweringModelOutput,
	SequenceClassifierOutput,
	TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
	ModelOutput,
	add_code_sample_docstrings,
	add_start_docstrings,
	add_start_docstrings_to_model_forward,
	logging,
	replace_return_docstrings,
)
from transformers import PretrainedConfig

logger = logging.get_logger(__name__)

class BaseBertConfig(PretrainedConfig):
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
		max_position_embeddings=512,
		type_vocab_size=2,
		initializer_range=0.02,
		layer_norm_eps=1e-12,
		pad_token_id=0,
		position_embedding_type="absolute",
		use_cache=True,
		classifier_dropout=None,
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
		self.max_position_embeddings = max_position_embeddings
		self.type_vocab_size = type_vocab_size
		self.initializer_range = initializer_range
		self.layer_norm_eps = layer_norm_eps
		self.position_embedding_type = position_embedding_type
		self.use_cache = use_cache
		self.classifier_dropout = classifier_dropout

class BaseBertEmbeddings(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
	"""
	input_ids: Encoded token inputs
	token_type_ids: artifact from NSP subtask of BERT. In this model the first sentence is the full conversation, while the 2nd sentence is a person???
	position_ids: position of each token in original sentence
	"""
	def forward(
		self,
		input_ids: torch.LongTensor,
		token_type_ids: Optional[torch.LongTensor] = None,
	) -> torch.Tensor:
		sentence_len = input_ids.size()[1]
		# Position ids are generated on the fly
		position_ids = torch.arange(sentence_len, dtype=torch.long, device=input_ids.device)
		position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
		# Token type 0 by default
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)
		# create embeddings and sum up
		inputs_embeds = self.word_embeddings(input_ids)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)
		position_embeddings = self.position_embeddings(position_ids)
		embeddings = inputs_embeds + token_type_embeddings + position_embeddings

		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings

class BaseBertSelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
			raise ValueError(
				f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
				f"heads ({config.num_attention_heads})"
			)

		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

		self.is_decoder = config.is_decoder

		self.output = BaseBertSelfOutput(config)

	def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.FloatTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		output_attentions: Optional[bool] = False,
	) -> Tuple[torch.Tensor]:
		'''
		Initialize key,value,query
		'''
		mixed_query_layer = self.query(hidden_states)
		# If this is instantiated as a cross-attention module, the keys
		# and values come from an encoder; the attention mask needs to be
		# such that the encoder's padding tokens are not attended to.
		is_cross_attention = encoder_hidden_states is not None
		if is_cross_attention and past_key_value is not None:
			# reuse k,v, cross_attentions
			key_layer = past_key_value[0]
			value_layer = past_key_value[1]
			attention_mask = encoder_attention_mask
		elif is_cross_attention:
			key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
			value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
			attention_mask = encoder_attention_mask
		elif past_key_value is not None:
			key_layer = self.transpose_for_scores(self.key(hidden_states))
			value_layer = self.transpose_for_scores(self.value(hidden_states))
			key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
			value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
		else:
			key_layer = self.transpose_for_scores(self.key(hidden_states))
			value_layer = self.transpose_for_scores(self.value(hidden_states))
		query_layer = self.transpose_for_scores(mixed_query_layer)
		if self.is_decoder:
			# if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
			# Further calls to cross_attention layer can then reuse all cross-attention
			# key/value_states (first "if" case)
			# if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
			# all previous decoder key/value_states. Further calls to uni-directional self-attention
			# can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
			# if encoder bi-directional self-attention `past_key_value` is always `None`
			past_key_value = (key_layer, value_layer)
		'''
		Scaled Dot-Product Attention
		'''
		# QK^T
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		# /sqrt(d_k)
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		if attention_mask is not None:
			attention_scores = attention_scores + attention_mask # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		# softmax()
		attention_probs = nn.functional.softmax(attention_scores, dim=-1)
		attention_probs = self.dropout(attention_probs)
		if head_mask is not None: attention_probs = attention_probs * head_mask # Mask heads if we want to
		# * V
		context_layer = torch.matmul(attention_probs, value_layer)
		'''
		
		'''
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(new_context_layer_shape)
		outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
		if self.is_decoder:
			outputs = outputs + (past_key_value,)

		attention_output = self.output(outputs[0], hidden_states)
		final_outputs = (attention_output,) + outputs[1:]  # add attentions if we output them
		return final_outputs

class BaseBertSelfOutput(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states

class BaseBertIntermediate(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		self.intermediate_act_fn = ACT2FN[config.hidden_act]

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states


class BaseBertOutput(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states

class BaseBertLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.chunk_size_feed_forward = config.chunk_size_feed_forward
		self.seq_len_dim = 1
		self.attention = BaseBertSelfAttention(config)
		self.is_decoder = config.is_decoder
		self.add_cross_attention = config.add_cross_attention
		if self.add_cross_attention:
			if not self.is_decoder:
				raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
			self.crossattention = BaseBertSelfAttention(config, position_embedding_type="absolute")
		self.intermediate = BaseBertIntermediate(config)
		self.output = BaseBertOutput(config)

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.FloatTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		output_attentions: Optional[bool] = False,
	) -> Tuple[torch.Tensor]:
		# decoder uni-directional self-attention cached key/values tuple is at positions 1,2
		self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
		self_attention_outputs = self.attention(
			hidden_states,
			attention_mask,
			head_mask,
			output_attentions=output_attentions,
			past_key_value=self_attn_past_key_value,
		)
		attention_output = self_attention_outputs[0]

		# if decoder, the last output is tuple of self-attn cache
		if self.is_decoder:
			outputs = self_attention_outputs[1:-1]
			present_key_value = self_attention_outputs[-1]
		else:
			outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

		cross_attn_present_key_value = None
		if self.is_decoder and encoder_hidden_states is not None:
			if not hasattr(self, "crossattention"):
				raise ValueError(
					f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
					" by setting `config.add_cross_attention=True`"
				)

			# cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
			cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
			cross_attention_outputs = self.crossattention(
				attention_output,
				attention_mask,
				head_mask,
				encoder_hidden_states,
				encoder_attention_mask,
				cross_attn_past_key_value,
				output_attentions,
			)
			attention_output = cross_attention_outputs[0]
			outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

			# add cross-attn cache to positions 3,4 of present_key_value tuple
			cross_attn_present_key_value = cross_attention_outputs[-1]
			present_key_value = present_key_value + cross_attn_present_key_value

		layer_output = apply_chunking_to_forward(
			self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
		)
		outputs = (layer_output,) + outputs

		# if decoder, return the attn key/values as the last output
		if self.is_decoder:
			outputs = outputs + (present_key_value,)

		return outputs

	def feed_forward_chunk(self, attention_output):
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		return layer_output

class BaseBertEncoder(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.layer = nn.ModuleList([BaseBertLayer(config) for _ in range(config.num_hidden_layers)])
		self.gradient_checkpointing = False

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.FloatTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = False,
		output_hidden_states: Optional[bool] = False,
		return_dict: Optional[bool] = True,
	) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
		all_hidden_states = () if output_hidden_states else None
		all_self_attentions = () if output_attentions else None
		all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

		if self.gradient_checkpointing and self.training:
			if use_cache:
				logger.warning_once(
					"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
				)
				use_cache = False

		next_decoder_cache = () if use_cache else None
		for i, layer_module in enumerate(self.layer):
			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

			layer_head_mask = head_mask[i] if head_mask is not None else None
			past_key_value = past_key_values[i] if past_key_values is not None else None

			if self.gradient_checkpointing and self.training:
				layer_outputs = self._gradient_checkpointing_func(
					layer_module.__call__,
					hidden_states,
					attention_mask,
					layer_head_mask,
					encoder_hidden_states,
					encoder_attention_mask,
					past_key_value,
					output_attentions,
				)
			else:
				layer_outputs = layer_module(
					hidden_states,
					attention_mask,
					layer_head_mask,
					encoder_hidden_states,
					encoder_attention_mask,
					past_key_value,
					output_attentions,
				)

			hidden_states = layer_outputs[0]
			if use_cache:
				next_decoder_cache += (layer_outputs[-1],)
			if output_attentions:
				all_self_attentions = all_self_attentions + (layer_outputs[1],)
				if self.config.add_cross_attention:
					all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		if not return_dict:
			return tuple(
				v
				for v in [
					hidden_states,
					next_decoder_cache,
					all_hidden_states,
					all_self_attentions,
					all_cross_attentions,
				]
				if v is not None
			)
		return BaseModelOutputWithPastAndCrossAttentions(
			last_hidden_state=hidden_states,
			past_key_values=next_decoder_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attentions,
			cross_attentions=all_cross_attentions,
		)

class BaseBertPooler(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.activation = nn.Tanh()

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		# We "pool" the model by simply taking the hidden state corresponding
		# to the first token.
		first_token_tensor = hidden_states[:, 0]
		pooled_output = self.dense(first_token_tensor)
		pooled_output = self.activation(pooled_output)
		return pooled_output

class BaseBertPreTrainedModel(PreTrainedModel):
	config_class = BaseBertConfig
	base_model_prefix = "bert"
	supports_gradient_checkpointing = True

	def _init_weights(self, module):
		"""Initialize the weights"""
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

class BaseBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BaseBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BaseBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BaseBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BaseBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BaseBertModel(BaseBertPreTrainedModel):
	"""

	The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
	cross-attention is added between the self-attention layers, following the architecture described in [Attention is
	all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
	Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

	To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
	to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
	`add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
	"""

	def __init__(self, config, add_pooling_layer=True):
		super().__init__(config)
		self.config = config

		self.embeddings = BaseBertEmbeddings(config)
		self.encoder = BaseBertEncoder(config)

		self.pooler = BaseBertPooler(config) if add_pooling_layer else None
		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.embeddings.word_embeddings

	def set_input_embeddings(self, value):
		self.embeddings.word_embeddings = value

	def _prune_heads(self, heads_to_prune):
		"""
		Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
		class PreTrainedModel
		"""
		for layer, heads in heads_to_prune.items():
			self.encoder.layer[layer].attention.prune_heads(heads)

	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		token_type_ids: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.Tensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.Tensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
		r"""
		encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
			the model is configured as a decoder.
		encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
			the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.
		past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
			Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

			If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
			don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
			`decoder_input_ids` of shape `(batch_size, sequence_length)`.
		use_cache (`bool`, *optional*):
			If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
			`past_key_values`).
		"""
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if self.config.is_decoder:
			use_cache = use_cache if use_cache is not None else self.config.use_cache
		else:
			use_cache = False

		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
			input_shape = input_ids.size()
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")

		batch_size, seq_length = input_shape
		device = input_ids.device if input_ids is not None else inputs_embeds.device

		# past_key_values_length
		past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

		if attention_mask is None:
			attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

		if token_type_ids is None:
			if hasattr(self.embeddings, "token_type_ids"):
				buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
				buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
				token_type_ids = buffered_token_type_ids_expanded
			else:
				token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

		# We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
		# ourselves in which case we just need to make it broadcastable to all heads.
		extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

		# If a 2D or 3D attention mask is provided for the cross-attention
		# we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
		if self.config.is_decoder and encoder_hidden_states is not None:
			encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
			encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
			if encoder_attention_mask is None:
				encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
			encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
		else:
			encoder_extended_attention_mask = None

		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

		embedding_output = self.embeddings(
			input_ids=input_ids,
			token_type_ids=token_type_ids
		)
		encoder_outputs = self.encoder(
			embedding_output,
			attention_mask=extended_attention_mask,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_extended_attention_mask,
			past_key_values=past_key_values,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		sequence_output = encoder_outputs[0]
		pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

		if not return_dict:
			return (sequence_output, pooled_output) + encoder_outputs[1:]

		return BaseModelOutputWithPoolingAndCrossAttentions(
			last_hidden_state=sequence_output,
			pooler_output=pooled_output,
			past_key_values=encoder_outputs.past_key_values,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
			cross_attentions=encoder_outputs.cross_attentions,
		)

class BaseBertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class BaseBertForNextSentencePrediction(BaseBertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.bert = BaseBertModel(config)
		self.cls = BaseBertOnlyNSPHead(config)

		# Initialize weights and apply final processing
		self.post_init()
	
	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		token_type_ids: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.Tensor] = None,
		labels: Optional[torch.Tensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		**kwargs,
	) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
			(see `input_ids` docstring). Indices should be in `[0, 1]`:

			- 0 indicates sequence B is a continuation of sequence A,
			- 1 indicates sequence B is a random sequence.

		Returns:

		Example:

		```python
		>>> from transformers import AutoTokenizer, BertForNextSentencePrediction
		>>> import torch

		>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
		>>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

		>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
		>>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
		>>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

		>>> outputs = model(**encoding, labels=torch.LongTensor([1]))
		>>> logits = outputs.logits
		>>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
		```
		"""

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		pooled_output = outputs[1]

		seq_relationship_scores = self.cls(pooled_output)

		next_sentence_loss = None
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

		if not return_dict:
			output = (seq_relationship_scores,) + outputs[2:]
			return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

		return NextSentencePredictorOutput(
			loss=next_sentence_loss,
			logits=seq_relationship_scores,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
