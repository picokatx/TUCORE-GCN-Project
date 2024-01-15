from difflib import ndiff
from data import TUCOREGCNDataset, TUCOREGCNDataloader
from tucore_gcn_transformers.tucore_gcn_bert_train import TUCOREGCNDialogREDataset
from models.BERT.tokenization import FullTokenizer
from tqdm import tqdm
import numpy as np
import datasets

class ANY:
    def __init__(self, *_types):
        self._types = _types

    def __eq__(self, other):
        return isinstance(other, self._types)

    def __repr__(self):
        return f"ANY({', '.join(_type.__name__ for _type in self._types)})"

def input_parity(entry_idx=None, split='dev', do_turn_mask_testing=False):
	tokenizer = FullTokenizer(vocab_file="../pre-trained_model/BERT/vocab.txt", do_lower_case=True)
	old_dataset = TUCOREGCNDataset("../datasets/DialogRE/", f"../datasets/DialogRE/{split}_BERT.pkl", 512, tokenizer, 36, "BERT")
	old_loader = TUCOREGCNDataloader(dataset=old_dataset, batch_size=12, shuffle=False, relation_num=36, max_length=512)
	old_data = old_loader.dataset.data
	new_dataset = TUCOREGCNDialogREDataset()
	new_data_generator = new_dataset._generate_examples(f'../datasets/DialogRE/{split}.json', split, 512, False, True, False)
	new_data = [data[1] for data in new_data_generator]
	#new_data = datasets.load_from_disk("../datasets/DialogRE/parity")[split if split!='dev' else "validation"]
	for idx in tqdm(range(0 if entry_idx==None else entry_idx, len(old_data) if entry_idx==None else entry_idx+1)):
		out = []
		cmp_all = ['label_ids', 'input_ids', 'input_mask', 'segment_ids', 'speaker_ids', 'mention_ids'] # graphs excluded, turn mask checked separately
		for cmp in cmp_all:
			out.append(f"------------|{cmp}|------------")
			a = "".join([chr(code) for code in (old_data[idx][cmp if not (cmp=="mention_ids" or cmp=="turn_masks") else ("mention_id" if cmp=="mention_ids" else "turn_mask")])])
			b = "".join([chr(code) for code in new_data[idx][cmp]])
			diff = ndiff(b,a)
			for i,s in enumerate(ndiff(b,a)):
				if s[0]==' ': continue
				elif s[0]=='-' or s[0]=='+':
					print(idx, cmp)
					#print(old_data[idx][cmp if not (cmp=="mention_ids" or cmp=="turn_masks") else ("mention_id" if cmp=="mention_ids" else "turn_mask")])
					#print(new_data[idx][cmp])
					if entry_idx!=None:
						print(u'Delete "{}" from position {}'.format(ord(s[-1]),i))
						print(u'Add "{}" to position {}'.format(ord(s[-1]),i))
					break
			'''
			for i,s in enumerate(ndiff(b,a)):
				if s[0]==' ': continue
				elif s[0]=='-':
					out.append(u'Delete "{}" from position {}'.format(ord(s[-1]),i))
				elif s[0]=='+':
					out.append(u'Add "{}" to position {}'.format(ord(s[-1]),i))
			'''
		'''
		graph_checks = ['graph_speaker', 'graph_dialog', 'graph_entity']
		for check in graph_checks:
			g_a = np.array(old_data[idx][check])
			g_b = np.array(new_data[idx][check])
			if (g_a.shape!=g_b.shape) or len(np.unique(g_a==g_b))!=1:
				print(idx, check)
				#print(g_a)
				#print(g_b)
		if do_turn_mask_testing and len(np.unique((old_data[idx]["turn_mask"] == new_data[idx]["turn_masks"])))!=1:
			print(idx, "turn_masks")
		'''
		'''	
		with open(f"data_diff{idx}.txt", 'w') as file:
			file.write("\n".join(out))
		'''