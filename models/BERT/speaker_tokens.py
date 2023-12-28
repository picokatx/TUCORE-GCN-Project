from dataclasses import dataclass
from transformers.models.bert.tokenization_bert import BertTokenizer
@dataclass
class SPEAKER_TOKENS():
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
		speaker_tokens = [entry[1] for idx, entry in list(filter(lambda x: x[1][1] if not x[1][0].startswith("__") else None, enumerate(SPEAKER_TOKENS.__dict__.items())))]
		return speaker_tokens.count(token)!=0
	def convert_speaker_to_id(token):
		speaker_tokens = [entry[1] for idx, entry in list(filter(lambda x: x[1][1] if not x[1][0].startswith("__") else None, enumerate(SPEAKER_TOKENS.__dict__.items())))]
		return speaker_tokens.index(token)

class SpeakerBertTokenizer(BertTokenizer):
	def __init__(self,
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
        **kwargs,):
		super().__init__(vocab_file,
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
        **kwargs)
		self.add_special_tokens(
			{
				'additional_special_tokens': [
					SPEAKER_TOKENS.SPEAKER_X,
					SPEAKER_TOKENS.SPEAKER_Y,
					SPEAKER_TOKENS.SPEAKER_1,
					SPEAKER_TOKENS.SPEAKER_2,
					SPEAKER_TOKENS.SPEAKER_3,
					SPEAKER_TOKENS.SPEAKER_4,
					SPEAKER_TOKENS.SPEAKER_5,
					SPEAKER_TOKENS.SPEAKER_6,
					SPEAKER_TOKENS.SPEAKER_7,
					SPEAKER_TOKENS.SPEAKER_8,
					SPEAKER_TOKENS.SPEAKER_9
				]
			},
			False
		)