from dataclasses import dataclass
from transformers.models.bert.tokenization_bert import BertTokenizer


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
		# Deprecated feature, best to change it later
		self.basic_tokenizer.never_split = set(self.all_special_tokens + list(self.speaker2id.keys()))
	def _tokenize(self, text, split_special_tokens=False):
		split_tokens = []
		if self.do_basic_tokenize:
			for token in self.basic_tokenizer.tokenize(text, never_split=set(self.all_special_tokens + list(self.speaker2id.keys()))):
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
