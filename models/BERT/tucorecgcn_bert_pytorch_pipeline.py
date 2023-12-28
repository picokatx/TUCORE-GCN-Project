import numpy as np

from transformers import Pipeline
from models.BERT.speaker_tokens import SPEAKER_TOKENS, SpeakerBertTokenizer
from models.BERT.tucoregcn_bert_pytorch_processor import Conversation, SpeakerRelation, Message

def softmax(outputs):
	maxes = np.max(outputs, axis=-1, keepdims=True)
	shifted_exp = np.exp(outputs - maxes)
	return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class ConversationalSequenceClassification(Pipeline):
	def _sanitize_parameters(self, **kwargs):
		preprocess_kwargs = {}
		if "second_text" in kwargs:
			preprocess_kwargs["second_text"] = kwargs["second_text"]
		return preprocess_kwargs, {}, {}

	def preprocess(self, conversation: Conversation):
		n_class = 36
		speaker_tokenizer = SpeakerBertTokenizer.from_pretrained('bert-base-uncased')
		test2 = conversation.build_inputs(speaker_tokenizer)[0]
		speaker_tokens = [entry[1] for idx, entry in list(filter(lambda x: x[1][1] if not x[1][0].startswith("__") else None, enumerate(SPEAKER_TOKENS.__dict__.items())))]
		sequence = speaker_tokenizer.tokenize(test2['dialog'])
		speaker_x = speaker_tokenizer.tokenize(test2['relation'].speaker_x)
		speaker_y = speaker_tokenizer.tokenize(test2['relation'].speaker_y)

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
		speaker_ids = [0] + input_speaker_ids + [0] + [0]*len(speaker_x) + [0] + [0]*len(speaker_y) + [0]
		mention_ids = [0] + input_mention_ids + [0] + [0]*len(speaker_x) + [0] + [0]*len(speaker_y) + [0]

		label_id = []
		for k in range(n_class):
			if k+1 in test2['relation'].rid:
				label_id.append(1)
			else:
				label_id.append(0)
		return tokens, input_ids, input_mask, segment_ids, speaker_ids, mention_ids

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