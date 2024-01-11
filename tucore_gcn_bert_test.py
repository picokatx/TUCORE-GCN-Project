from difflib import ndiff
from data import TUCOREGCNDataset, TUCOREGCNDataloader
from tucore_gcn_transformers.tucore_gcn_bert_train import TUCOREGCNDialogREDataset
from models.BERT.tokenization import FullTokenizer
from tqdm import tqdm
import numpy as np

def input_parity(entry_idx=None, split='dev', do_turn_mask_testing=False):
	tokenizer = FullTokenizer(vocab_file="../pre-trained_model/BERT/vocab.txt", do_lower_case=True)
	old_dataset = TUCOREGCNDataset("../datasets/DialogRE/", f"../datasets/DialogRE/{split}_BERT.pkl", 512, tokenizer, 36, "BERT")
	old_loader = TUCOREGCNDataloader(dataset=old_dataset, batch_size=12, shuffle=False, relation_num=36, max_length=512)
	old_data = old_loader.dataset.data
	new_dataset = TUCOREGCNDialogREDataset()
	new_data_generator = new_dataset._generate_examples(f'../datasets/DialogRE/{split}.json', split, 512, False, True, False)
	new_data = [data[1] for data in new_data_generator]
	for idx in tqdm(range(0 if entry_idx==None else entry_idx, len(old_data) if entry_idx==None else entry_idx+1)):
		out = []
		cmp_all = list(new_data[idx].keys())[1:-2] # graphs excluded, turn mask checked separately

		for cmp in cmp_all:
			out.append(f"------------|{cmp}|------------")
			a = "".join([chr(code) for code in (old_data[idx][cmp if not (cmp=="mention_ids" or cmp=="turn_masks") else ("mention_id" if cmp=="mention_ids" else "turn_mask")])])
			b = "".join([chr(code) for code in new_data[idx][cmp]])
			diff = ndiff(b,a)
			for i,s in enumerate(ndiff(b,a)):
				if s[0]==' ': continue
				elif s[0]=='-' or s[0]=='+':
					print(idx, cmp)
					print(a)
					print(b)
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
		if do_turn_mask_testing and len(np.unique((old_data[idx]["turn_mask"] == new_data[idx]["turn_masks"])))!=1:
			print(idx)
		'''	
		with open(f"data_diff{idx}.txt", 'w') as file:
			file.write("\n".join(out))
		'''