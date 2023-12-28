import numpy as np

from transformers import Pipeline
from models.BERT.speaker_tokens import SPEAKER_TOKENS, SpeakerBertTokenizer
from models.BERT.tucoregcn_bert_pytorch_processor import Conversation, SpeakerRelation, Message
from collections import defaultdict
from itertools import permutations
import dgl

def make_speaker_infor(speaker_id, mention_id):
	tmp = defaultdict(set)
	for i in range(1, len(speaker_id)):
		if speaker_id[i] == 0:
			break
		tmp[speaker_id[i]].add(mention_id[i])
	
	speaker_infor = dict()
	for k, va in tmp.items():
		speaker_infor[k] = list(va)
	return speaker_infor

def make_entity_edges_infor(input_ids, mention_id):
	entity_edges_infor = {'h':[], 't':[]}
	head_mention_id = max(mention_id) - 1
	tail_mention_id = max(mention_id)
	head = list()
	tail = list()
	for i in range(len(mention_id)):
		if mention_id[i] == head_mention_id:
			head.append(input_ids[i])

	for i in range(len(mention_id)):
		if mention_id[i] == tail_mention_id:
			tail.append(input_ids[i])
	
	for i in range(len(input_ids) - len(head)):
		if input_ids[i:i+len(head)] == head:
			entity_edges_infor['h'].append(mention_id[i])
	
	for i in range(len(input_ids) - len(tail)):
		if input_ids[i:i+len(tail)] == tail:
			entity_edges_infor['t'].append(mention_id[i])
	
	return entity_edges_infor

def create_graph(speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id):
	d = defaultdict(list)
	used_mention = set()

	# add speaker edges
	for _, mentions in speaker_infor.items():
		for h, t in permutations(mentions, 2):
			d[('node', 'speaker', 'node')].append((h, t))
			used_mention.add(h)
			used_mention.add(t)
	
	if d[('node', 'speaker', 'node')] == []:
		d[('node', 'speaker', 'node')].append((1, 0))
		used_mention.add(1)
		used_mention.add(0)
	

	# add dialog edges
	for i in range(1, turn_node_num+1):
		d[('node', 'dialog', 'node')].append((i, 0))
		d[('node', 'dialog', 'node')].append((0, i))
		used_mention.add(i)
		used_mention.add(0)
	if d[('node', 'dialog', 'node')] == []:
		d[('node', 'dialog', 'node')].append((1, 0))
		used_mention.add(1)
		used_mention.add(0)

	# add entity edges
	for mention in entity_edges_infor['h']:
		d[('node', 'entity', 'node')].append((head_mention_id, mention))
		d[('node', 'entity', 'node')].append((mention, head_mention_id))
		used_mention.add(head_mention_id)
		used_mention.add(mention)
	
	for mention in entity_edges_infor['t']:
		d[('node', 'entity', 'node')].append((tail_mention_id, mention))
		d[('node', 'entity', 'node')].append((mention, tail_mention_id))
		used_mention.add(tail_mention_id)
		used_mention.add(mention)
	
	if entity_edges_infor['h'] == []:
		d[('node', 'entity', 'node')].append((head_mention_id, 0))
		d[('node', 'entity', 'node')].append((0, head_mention_id))
		used_mention.add(head_mention_id)
		used_mention.add(0)
	
	if entity_edges_infor['t'] == []:
		d[('node', 'entity', 'node')].append((tail_mention_id, 0))
		d[('node', 'entity', 'node')].append((0, tail_mention_id))
		used_mention.add(tail_mention_id)
		used_mention.add(0)

	graph = dgl.heterograph(d)

	return graph, used_mention
def softmax(outputs):
	maxes = np.max(outputs, axis=-1, keepdims=True)
	shifted_exp = np.exp(outputs - maxes)
	return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

def mention2mask(mention_id, old_behaviour=False):
	slen = len(mention_id)
	mask = []
	if old_behaviour:
		turn_mention_ids = [i for i in range(1, np.max(mention_id)-1)] #-1
	else:
		turn_mention_ids = [i for i in range(1, np.max(mention_id)+1)] #-1
	print(turn_mention_ids)
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

class ConversationalSequenceClassificationPipeline(Pipeline):
	def _sanitize_parameters(self, **kwargs):
		preprocess_kwargs = {}
		if "tokenizer" in kwargs:
			preprocess_kwargs["speaker_tokenizer"] = kwargs["tokenizer"]
		return preprocess_kwargs, {}, {}

	def preprocess(self, conversation: Conversation, speaker_tokenizer):
		n_class = 36
		max_seq_length=512
		old_behaviour = True
		inputs = conversation.build_inputs(speaker_tokenizer)[0]
		sequence = speaker_tokenizer.tokenize(inputs['dialog'])
		speaker_x = speaker_tokenizer.tokenize(inputs['relation'].speaker_x)
		speaker_y = speaker_tokenizer.tokenize(inputs['relation'].speaker_y)

		tokens = ["[CLS]"] + sequence + ["[SEP]"] + speaker_x + ["[SEP]"] + speaker_y + ["[SEP]"]
		input_ids = speaker_tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] + [1]*len(sequence) + [1] + [1]*len(speaker_x) + [1] + [1]*len(speaker_y) + [0]
		segment_ids = [0] + [0]*len(sequence) + [0] + [1]*len(speaker_x) + [1] + [1]*len(speaker_y) + [1]

		input_speaker_ids = []
		input_mention_ids = []
		current_speaker_id = 0
		current_speaker_idx = 0
		for token in sequence:
			if SPEAKER_TOKENS.is_speaker(token):
				current_speaker_id = SPEAKER_TOKENS.convert_speaker_to_id(token)
				current_speaker_idx+=1
			input_speaker_ids.append(current_speaker_id)
			input_mention_ids.append(current_speaker_idx)
		if old_behaviour:
			speaker_ids = [0] + input_speaker_ids + [0] + [SPEAKER_TOKENS.convert_speaker_to_id(inputs['relation'].speaker_x)]*len(speaker_x) + [0] + [SPEAKER_TOKENS.convert_speaker_to_id(inputs['relation'].speaker_y)]*len(speaker_y) + [0]
			mention_ids = [0] + input_mention_ids + [0] + [current_speaker_idx+1]*len(speaker_x) + [0] + [current_speaker_idx+2]*len(speaker_y) + [0]
		else:
			speaker_ids = [0] + input_speaker_ids + [0] + [0]*len(speaker_x) + [0] + [0]*len(speaker_y) + [0]
			mention_ids = [0] + input_mention_ids + [0] + [0]*len(speaker_x) + [0] + [0]*len(speaker_y) + [0]

		label_id = []
		for k in range(n_class):
			if k+1 in inputs['relation'].rid:
				label_id.append(1)
			else:
				label_id.append(0)
		while len(input_ids) < max_seq_length:
			tokens.append('[PAD]')
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)
			speaker_ids.append(0)
			mention_ids.append(0)

		turn_masks = mention2mask(np.array(mention_ids), old_behaviour)
		
		speaker_infor = make_speaker_infor(speaker_ids, mention_ids)
		turn_node_num = max(mention_ids) - 2
		head_mention_id = max(mention_ids) - 1
		tail_mention_id = max(mention_ids)
		entity_edges_infor = make_entity_edges_infor(input_ids, mention_ids)
		graph, used_mention = create_graph(speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id)
		assert len(used_mention) == (max(mention_ids) + 1)

		return (
			np.array(tokens),
			np.array(input_ids),
			np.array(input_mask),
			np.array(segment_ids),
			np.array(speaker_ids),
			np.array(mention_ids),
			turn_masks,
			graph
		)

	def _forward(self, model_inputs):
		return self.model(**model_inputs)

	def postprocess(self, model_outputs):
		logits = model_outputs.logits[0].numpy()
		probabilities = softmax(logits)

		best_class = np.argmax(probabilities)
		label = self.model.config.id2label[best_class]
		score = probabilities[best_class].item()
		logits = logits.tolist()
		return {"label": label, "score": score, "logits": logits}