from difflib import ndiff
from tucore_gcn_old_processor import TUCOREGCNDataset, TUCOREGCNDataloader
from tucore_gcn_old_tokenizer import FullTokenizer
from tucore_gcn_bert_train import TUCOREGCNDialogREDataset
from tucore_gcn_bert_tokenizer import SPEAKER_TOKENS, SpeakerBertTokenizer
from tqdm import tqdm
import numpy as np
import datasets
import unittest
from parameterized import parameterized
import math
import pickle
import os
from transformers.testing_utils import (
    TOKEN,
    USER,
    CaptureLogger,
    RequestCounter,
    backend_empty_cache,
    is_pipeline_test,
    is_staging_test,
    nested_simplify,
    require_tensorflow_probability,
    require_tf,
    require_torch,
    require_torch_accelerator,
    require_torch_or_tf,
    slow,
    torch_device,
)

class ANY:
    def __init__(self, *_types):
        self._types = _types

    def __eq__(self, other):
        return isinstance(other, self._types)

    def __repr__(self):
        return f"ANY({', '.join(_type.__name__ for _type in self._types)})"

def get_entry(idx, save_file):
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                print('load preprocessed data from {}.'.format(save_file))
                return info['data'][idx]

class TUCOREGCNBertTokenizerTest(unittest.TestCase):
    speaker_tokenizer: SpeakerBertTokenizer = SpeakerBertTokenizer.from_pretrained(
        "bert-base-uncased"
    )
    @parameterized.expand(
        [
            ("{entity_1}:", ["{", "entity", "_", "1", "}", ":"]), # tokenization breaks if colon isn't pre_padded
            ("Café", ["cafe"]), # normalize accents by default
            ("Monika", ["mon", "##ika"]),
            ("{entity_1} :", ["{entity_1}", ":"]),
            ("speaker 1:", ["speaker", "1", ":"]),
            ("{entity_1} : Hello World!", ["{entity_1}", ":", "hello", "world", "!"]),
            ("{entity_1} : Hello World!\nspeaker 2: Hi!", ["{entity_1}", ":", "hello", "world", "!", "speaker", "2", ":", "hi", "!"]),
            ("[PAD] [CLS] [UNK] [SEP] [MASK] Location", ["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]", "location"]),
        ]
    )
    def test_tokenize(self, s, expected):
        self.assertEqual(self.speaker_tokenizer._tokenize(s), expected)
    @parameterized.expand(
        [
            (2, SPEAKER_TOKENS.ENTITY_1),
            (3, SPEAKER_TOKENS.ENTITY_2),
            (0, "[PAD]"),
            (100, "[UNK]"),
            (101, "[CLS]"),
            (102, "[SEP]"),
            (103, "[MASK]"),
            (3295, "location"),
        ]
    )
    def test_convert_id_to_token(self, idx, expected):
        self.assertEqual(self.speaker_tokenizer._convert_id_to_token(idx), expected)
    @parameterized.expand(
        [
            (SPEAKER_TOKENS.ENTITY_1, 2),
            (SPEAKER_TOKENS.ENTITY_2, 3),
            ("[PAD]", 0),
            ("flowey", 100),
            ("[UNK]", 100),
            ("[CLS]", 101),
            ("[SEP]", 102),
            ("[MASK]", 103),
            ("location", 3295),
        ]
    )
    def test_convert_token_to_id(self, token, expected):
        self.assertEqual(self.speaker_tokenizer._convert_token_to_id(token), expected)

    @parameterized.expand(
        [
            (SPEAKER_TOKENS.ENTITY_1, True),
            (SPEAKER_TOKENS.ENTITY_2, True),
            ("control", False),
        ]
    )
    def test_is_speaker(self, token, expected):
        self.assertEqual(self.speaker_tokenizer.is_speaker(token), expected)

    @parameterized.expand(
        [
            (SPEAKER_TOKENS.ENTITY_1, 11),
            (SPEAKER_TOKENS.ENTITY_2, 12),
            ("control", "control"),
        ]
    )
    def test_convert_speaker_to_id(self, token, expected):
        self.assertEqual(self.speaker_tokenizer.convert_speaker_to_id(token), expected)

class DevEntryParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for model_type in ['bert', 'roberta']:
            setattr(cls, f"new_dataset_{model_type}", TUCOREGCNDialogREDataset(model_type=model_type))
            for split in ['dev']:
                if os.path.exists(f"test_data/{split}_{model_type}_testing.pkl"):
                    with open(file=f"test_data/{split}_{model_type}_testing.pkl", mode='rb') as f:
                        setattr(cls, f"old_dataset_{split}_{model_type}", pickle.load(f)['data'])

    @parameterized.expand(
        [
            (idx, 'bert') for idx in range(0,1914)
        ]
    )
    def test_all_entries(self, idx=0, model_type='bert'):
        old = getattr(self, f"old_dataset_dev_{model_type}")[idx]
        _, new = getattr(self, f"new_dataset_{model_type}").generate_example(idx=idx, split='dev', filepath=f"./test_data/dev.json", shuffle=False)
        out = []
        cmp_all = [
            "label_ids",
            "input_ids",
            "input_mask",
            "segment_ids",
            "speaker_ids",
            "mention_ids",
        ]  # graphs excluded, turn mask checked separately
        for cmp in cmp_all:
            out.append(f"------------|{cmp}|------------")
            a = "".join(
                [
                    chr(code)
                    for code in (
                        old[
                            cmp
                            if not (cmp == "mention_ids" or cmp == "turn_masks")
                            else ("mention_id" if cmp == "mention_ids" else "turn_mask")
                        ]
                    )
                ]
            )
            b = "".join([chr(code) for code in new[cmp]])
            diff = []
            for i, s in enumerate(ndiff(b, a)):
                if s[0] == " ":
                    continue
                elif s[0] == "-" or s[0] == "+":
                    diff.append((idx, cmp))
            self.assertTrue(len(diff)==0)

class TestEntryParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for model_type in ['bert', 'roberta']:
            setattr(cls, f"new_dataset_{model_type}", TUCOREGCNDialogREDataset(model_type=model_type))
            for split in ['test']:
                if os.path.exists(f"test_data/{split}_{model_type}_testing.pkl"):
                    with open(file=f"test_data/{split}_{model_type}_testing.pkl", mode='rb') as f:
                        setattr(cls, f"old_dataset_{split}_{model_type}", pickle.load(f)['data'])

    @parameterized.expand(
        [
            (idx, 'bert') for idx in range(0,1862)
        ]
    )
    def test_all_entries(self, idx=0, model_type='bert'):
        old = getattr(self, f"old_dataset_test_{model_type}")[idx]
        _, new = getattr(self, f"new_dataset_{model_type}").generate_example(idx=idx, split='test', filepath=f"./test_data/test.json", shuffle=False)
        out = []
        cmp_all = [
            "label_ids",
            "input_ids",
            "input_mask",
            "segment_ids",
            "speaker_ids",
            "mention_ids",
        ]  # graphs excluded, turn mask checked separately
        for cmp in cmp_all:
            out.append(f"------------|{cmp}|------------")
            a = "".join(
                [
                    chr(code)
                    for code in (
                        old[
                            cmp
                            if not (cmp == "mention_ids" or cmp == "turn_masks")
                            else ("mention_id" if cmp == "mention_ids" else "turn_mask")
                        ]
                    )
                ]
            )
            b = "".join([chr(code) for code in new[cmp]])
            diff = []
            for i, s in enumerate(ndiff(b, a)):
                if s[0] == " ":
                    continue
                elif s[0] == "-" or s[0] == "+":
                    diff.append((idx, cmp))
            self.assertTrue(len(diff)==0, f"matching failed at {cmp}")

class TrainEntryParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for model_type in ['bert', 'roberta']:
            setattr(cls, f"new_dataset_{model_type}", TUCOREGCNDialogREDataset(model_type=model_type))
            for split in ['train']:
                if os.path.exists(f"test_data/{split}_{model_type}_testing.pkl"):
                    with open(file=f"test_data/{split}_{model_type}_testing.pkl", mode='rb') as f:
                        setattr(cls, f"old_dataset_{split}_{model_type}", pickle.load(f)['data'])

    @parameterized.expand(
        [
            (idx, 'bert') for idx in range(0,5997)
        ]
    )
    def test_all_entries(self, idx=0, model_type='bert'):
        old = getattr(self, f"old_dataset_train_{model_type}")[idx]
        _, new = getattr(self, f"new_dataset_{model_type}").generate_example(idx=idx, split='train', filepath=f"./test_data/train.json", shuffle=False)
        out = []
        cmp_all = [
            "label_ids",
            "input_ids",
            "input_mask",
            "segment_ids",
            "speaker_ids",
            "mention_ids",
        ]  # graphs excluded, turn mask checked separately
        for cmp in cmp_all:
            out.append(f"------------|{cmp}|------------")
            a = "".join(
                [
                    chr(code)
                    for code in (
                        old[
                            cmp
                            if not (cmp == "mention_ids" or cmp == "turn_masks")
                            else ("mention_id" if cmp == "mention_ids" else "turn_mask")
                        ]
                    )
                ]
            )
            b = "".join([chr(code) for code in new[cmp]])
            diff = []
            for i, s in enumerate(ndiff(b, a)):
                if s[0] == " ":
                    continue
                elif s[0] == "-" or s[0] == "+":
                    diff.append((idx, cmp))
            self.assertTrue(len(diff)==0, f"matching failed at {cmp}")







def input_parity(entry_idx=None, split="dev", do_turn_mask_testing=False, model_type='BERT', tokenizer=None):
    if tokenizer==None: return
    old_dataset = TUCOREGCNDataset(
        "../datasets/DialogRE/",
        f"../datasets/DialogRE/{split}_{str.lower(model_type)}.pkl",
        512,
        tokenizer,
        36,
        str.lower(model_type),
    )
    old_loader = TUCOREGCNDataloader(
        dataset=old_dataset,
        batch_size=12,
        shuffle=False,
        relation_num=36,
        max_length=512,
    )
    old_data = old_loader.dataset.data
    new_dataset = TUCOREGCNDialogREDataset()
    new_data_generator = new_dataset._generate_examples(
        f"../datasets/DialogRE/{split}.json", split, 512, True, False, str.lower(model_type)
    )
    new_data = [data[1] for data in new_data_generator]
    # new_data = datasets.load_from_disk("../datasets/DialogRE/parity")[split if split!='dev' else "validation"]
    print("Now doing parity test")
    for idx in tqdm(
        range(
            0 if entry_idx == None else entry_idx,
            len(old_data) if entry_idx == None else entry_idx + 1,
        )
    ):
        out = []
        cmp_all = [
            "label_ids",
            "input_ids",
            "input_mask",
            "segment_ids",
            "speaker_ids",
            "mention_ids",
        ]  # graphs excluded, turn mask checked separately
        for cmp in cmp_all:
            out.append(f"------------|{cmp}|------------")
            a = "".join(
                [
                    chr(code)
                    for code in (
                        old_data[idx][
                            cmp
                            if not (cmp == "mention_ids" or cmp == "turn_masks")
                            else ("mention_id" if cmp == "mention_ids" else "turn_mask")
                        ]
                    )
                ]
            )
            b = "".join([chr(code) for code in new_data[idx][cmp]])
            diff = ndiff(b, a)
            for i, s in enumerate(ndiff(b, a)):
                if s[0] == " ":
                    continue
                elif s[0] == "-" or s[0] == "+":
                    print(idx, cmp)
                    # print(old_data[idx][cmp if not (cmp=="mention_ids" or cmp=="turn_masks") else ("mention_id" if cmp=="mention_ids" else "turn_mask")])
                    # print(new_data[idx][cmp])
                    if entry_idx != None:
                        print('Delete "{}" from position {}'.format(ord(s[-1]), i))
                        print('Add "{}" to position {}'.format(ord(s[-1]), i))
                    break
            """for i,s in enumerate(ndiff(b,a)):
                if s[0]==' ': continue
                elif s[0]=='-':
                    out.append(u'Delete "{}" from position {}'.format(ord(s[-1]),i))
                elif s[0]=='+':
                    out.append(u'Add "{}" to position {}'.format(ord(s[-1]),i))"""
        if do_turn_mask_testing and len(np.unique((old_data[idx]["turn_mask"] == new_data[idx]["turn_masks"])))!=1:
            print(idx, "turn_masks")
        """
        graph_checks = ['graph_speaker', 'graph_dialog', 'graph_entity']
        for check in graph_checks:
            g_a = np.array(old_data[idx][check])
            g_b = np.array(new_data[idx][check])
            if (g_a.shape!=g_b.shape) or len(np.unique(g_a==g_b))!=1:
                print(idx, check)
                #print(g_a)
                #print(g_b)
        """
        """    
        with open(f"data_diff{idx}.txt", 'w') as file:
            file.write("\n".join(out))
        """
