import os
import pickle
import random
import time
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import transformers
from transformers import AutoTokenizer
from collections import Counter

from utils.data_utils import load_args_from_parser

args = load_args_from_parser()
single_subtable_on = args.single_subtable_on
shortcut_name = args.shortcut_name
# args.encoding = "llm"
# shortcut_name = "Qwen/Qwen3-Embedding-8B"
tokenizer = AutoTokenizer.from_pretrained(shortcut_name) 
# https://huggingface.co/Qwen/Qwen3-Embedding-8B
seq_len_limit = 32000

# print(tokenizer.cls_token_id)
# print(tokenizer.sep_token_id)
# print(tokenizer.pad_token_id)
# print(tokenizer.mask_token_id)

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def pad_sequence_left(sequences, padding_value=0):
    """
    Left-pads a list of 1D torch tensors to the same length.

    Args:
        sequences (List[Tensor]): each tensor is 1D.
        padding_value (int): value to pad with.

    Returns:
        Tensor: shape (max_len, batch)
    """
    max_len = max(seq.size(0) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), padding_value, dtype=seq.dtype)
            padded_seq = torch.cat((pad_tensor, seq), dim=0)  # left pad
        else:
            padded_seq = seq
        padded.append(padded_seq)
    return torch.stack(padded, dim=1)  # shape: (max_len, batch)

def retrieve(input_ids, cls_idx):

    # set padding positions as 0
    # print( "-" * 8 + "inputs_id", input_ids)
    first_filter = input_ids == tokenizer.pad_token_id
    # print(tokenizer.pad_token_id)
    # print( "-" * 8 + "bool", first_filter)
    first_zero = (first_filter == 0).float().argmax(dim=1)
    # print( "-" * 8 + "index", first_zero)
    for i, idx in enumerate(first_zero):
        first_filter[i, :idx] = 0
    # print("-" * 8 + "final", first_filter)
    first_filter = torch.nonzero(first_filter)

    # for ele in first_filter:
    #     print( "-" * 8 + "final (index)", ele)

    # we need to filter out unannotated (context) columns
    output = []
    # print("-" * 8 + "cls_idx", cls_idx)
    for sample_idx, token_idx in enumerate(cls_idx):
        # print("-" * 8 + "token_idx", token_idx)
        # left padding
        token_idx = [ -(pos+1) for pos in token_idx]
        # print("-" * 8 + "token_idx(changed)", token_idx)
        mask = (first_filter[:, 0] == sample_idx)
        rows = first_filter[mask][token_idx]
        output.append(rows)
    return torch.cat(output)


def collate_fn(samples):

    input_ids = pad_sequence_left(
        [sample["input_ids"] for sample in samples], padding_value=tokenizer.pad_token_id
    )

    attention_mask = pad_sequence_left(
        [sample["attention_mask"] for sample in samples], padding_value= 0
    )

    label = torch.cat([torch.tensor(sample["label"]) for sample in samples])

    tbnames = []
    topic = []
    for sample in samples:
        for _ in range(len(sample["label"])):
            tbnames.append(sample["tbname"])
            topic.append(sample["tbname"].split('_')[0])

    cls_idx = [sample["cls_idx"] for sample in samples]
    cls_idx = retrieve(input_ids.T, cls_idx)

    batch = {"input_ids": input_ids.T, "attention_mask": attention_mask.T, "cls_idx": cls_idx, "label": label, "topic": topic, "tbname": tbnames}
    return batch


def collate_test_fn(samples):

    input_ids = pad_sequence_left(
        [sample["input_ids"] for sample in samples], padding_value=tokenizer.pad_token_id
    )

    attention_mask = pad_sequence_left(
        [sample["attention_mask"] for sample in samples], padding_value= 0
    )

    cls_idx = [sample["cls_idx"] for sample in samples]
    col_nos = [sample["col_nos"] for sample in samples]
    tbnames = [sample["tbname"] for sample in samples]
    # Added: cuts based on table columns for later inference
    cuts = [sample["cut"] for sample in samples]

    cls_idx = retrieve(input_ids.T, cls_idx)

    batch = {"input_ids": input_ids.T, "attention_mask": attention_mask.T, "cls_idx": cls_idx, "tbname": tbnames, "col_nos": col_nos, "cut": cuts}
    return batch

class CTATestDataset(Dataset):

    def __init__(self,
                 data_dict: dict,
                 table_dict: dict,
                 split: str,
                 args):

        self.idx2label = data_dict['idx2label']
        self.label2idx = data_dict['label2idx']
        self.num_label = max(self.idx2label.keys()) + 1

        assert split in data_dict
        self.split = split
        self.data_dict = data_dict[split]
        self.table_dict = table_dict
        self.data_list = []

        self.cls_token = tokenizer.cls_token_id
        self.sep_token = tokenizer.sep_token_id

        self.max_cell_len = args.max_cell_len
        self.max_col = args.max_col
        self.num_row_per_sample = args.num_row_per_sample

    def generate_epoch(self):

        num_row_per_sample = self.num_row_per_sample
        max_col = self.max_col

        self.data_list = []

        for tbname, data in self.data_dict.items():

            table = self.table_dict[tbname]

            if len(data['col_idx']) <= max_col:
                col_idx = sorted([ int(idx) for idx in data['col_idx'] ])
                col_idxs = copy.deepcopy([col_idx])
            else:
                col_idx_before = sorted([int(idx) for i, idx in enumerate(data['col_idx']) if i < max_col])
                col_idx_after = sorted([int(idx) for i, idx in enumerate(data['col_idx']) if i >= max_col])
                col_idxs = copy.deepcopy([col_idx_before, col_idx_after])
                assert len(col_idx_after) <= max_col

            row_cnt = len(table)
            col_cnt = len(table[0])

            for cut, col_idx in enumerate(col_idxs):

                col_flag = True if col_cnt <= max_col else False

                tb_row_nos = np.arange(0, row_cnt).tolist()

                for i in range( len(tb_row_nos) // num_row_per_sample ):

                    if col_flag:
                        col_nos = np.arange(0, col_cnt).tolist()
                        cls_idx = col_idx
                    else:
                        population = np.setdiff1d(np.arange(0, col_cnt), col_idx)
                        col_nos = np.sort( np.append(col_idx, np.random.choice(population, size=max_col-len(col_idx), replace=False)))
                        cls_idx = np.where(np.isin(col_nos, np.array(col_idx)))[0].tolist()
                        col_nos = col_nos.tolist()

                    row_nos = tb_row_nos[ i*num_row_per_sample : (i+1)*num_row_per_sample ]

                    assert len(row_nos) == num_row_per_sample
                    assert len(cls_idx) == len(col_idx)

                    self.data_list.append(copy.deepcopy([tbname, row_nos, col_nos, cls_idx, col_idx, cut]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        item = {}

        tbname, row_nos, col_nos, cls_idx, col_idx, cut = self.data_list[idx]

        item['col_idx'] = col_idx

        input_ids = []
        table = np.array(self.table_dict[tbname], dtype=object)
        for col in [ col for col in table[row_nos, :][:, col_nos].T ][::-1]:
            # bert ele: [CLS] text [SEP] => ele[1:-1]: text
            # qwen ele: text '<|endoftext|>
            for ele in col:
                if len(ele[:-1]) <= self.max_cell_len:
                    input_ids.extend(ele)
                else:
                    input_ids.extend(ele[:self.max_cell_len] + [ tokenizer.pad_token_id ] )
        assert len(input_ids) <= seq_len_limit

        item['input_ids'] = torch.tensor(input_ids, dtype=torch.int)
        item['attention_mask'] = torch.tensor([1] * len(input_ids), dtype=torch.int)

        item['cls_idx'] = cls_idx
        item['col_nos'] = col_nos
        item['tbname'] = tbname
        item['cut'] = cut

        return item

class CTADataset(Dataset):

    def __init__(self,
                 data_dict: dict,
                 table_dict: dict,
                 split: str,
                 args):

        self.idx2label = data_dict['idx2label']
        self.label2idx = data_dict['label2idx']

        self.num_label = max(self.idx2label.keys()) + 1

        assert split in data_dict
        self.split = split
        self.data_dict = data_dict[split]
        self.table_dict = table_dict
        self.data_list = []

        self.cls_token = tokenizer.cls_token_id
        self.sep_token = tokenizer.sep_token_id

        self.max_cell_len = args.max_cell_len
        self.max_col = args.max_col
        self.max_row = args.max_row
        self.num_row_per_sample = args.num_row_per_sample

    def generate_epoch(self):

        max_col = self.max_col
        max_row = self.max_row
        num_row_per_sample = self.num_row_per_sample

        self.data_list = []

        for tbname, data in self.data_dict.items():

            if len(data['label']) == 0:
                continue

            if len(data['col_idx']) == 0:
                continue

            table = self.table_dict[tbname]

            combos = [(l, int(idx)) for l, idx in zip(data['label'], data['col_idx'])]

            if len(data['col_idx']) <= max_col:
                chose_combos = sorted(combos, key=lambda x: x[1])
            else:
                chose_combos = sorted(random.sample(combos, max_col), key=lambda x: x[1])

            label = []
            col_idx = []
            for (l, idx) in chose_combos:
                label.append(l)
                col_idx.append(idx)

            # the code above solves the case where there are more than 10 annotated columns

            row_cnt = len(table)
            col_cnt = len(table[0])

            col_flag = True if col_cnt <= max_col else False
            row_flag = True if row_cnt <= max_row else False

            if self.split == 'train':
                if row_flag:
                    tb_row_nos = np.random.choice(np.arange(0,row_cnt), size=row_cnt, replace=False).tolist()
                else:
                    tb_row_nos = np.random.choice(np.arange(0,row_cnt), size=max_row, replace=False).tolist()
                assert len(tb_row_nos) // num_row_per_sample <= max_row / num_row_per_sample
            else:
                tb_row_nos = np.arange(0, row_cnt).tolist()

            # the code above picking up the rows

            for i in range( len(tb_row_nos) // num_row_per_sample ):

                if col_flag:
                    col_nos = np.arange(0, col_cnt).tolist()
                    cls_idx = col_idx
                    # use the orignal indices when the number of columns is smaller than the max number of columns (threshold)
                else:
                    population = np.setdiff1d(np.arange(0, col_cnt), col_idx)
                    col_nos = np.sort( np.append(col_idx, np.random.choice(population, size=max_col-len(col_idx), replace=False)))
                    cls_idx = np.where(np.isin(col_nos, np.array(col_idx)))[0].tolist()
                    col_nos = col_nos.tolist()

                row_nos = tb_row_nos[ i*num_row_per_sample : (i+1)*num_row_per_sample ]

                assert len(row_nos) == num_row_per_sample
                assert len(cls_idx) == len(label)

                self.data_list.append(copy.deepcopy([tbname, row_nos, col_nos, cls_idx, label]))

                if single_subtable_on and self.split == "train":
                    break

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        item = {}

        tbname, row_nos, col_nos, cls_idx, label = self.data_list[idx]

        input_ids = []
        table = np.array(self.table_dict[tbname], dtype=object)
        for col in [ col for col in table[row_nos, :][:, col_nos].T ][::-1]:
            # bert ele: [CLS] text [SEP] => ele[1:-1]: text
            # qwen ele: text '<|endoftext|>
            for ele in col:
                if len(ele[:-1]) <= self.max_cell_len:
                    input_ids.extend(ele)
                else:
                    input_ids.extend(ele[:self.max_cell_len] + [ tokenizer.pad_token_id ] )
        assert len(input_ids) <= seq_len_limit

        item['input_ids'] = torch.tensor(input_ids, dtype=torch.int)
        item['attention_mask'] = torch.tensor([1] * len(input_ids), dtype=torch.int)

        item['cls_idx'] = cls_idx
        item['label'] = label
        item['col_nos'] = col_nos
        item['tbname'] = tbname

        return item

class CPATestDataset(Dataset):

    def __init__(self,
                 data_dict: dict,
                 table_dict: dict,
                 split: str,
                 args):

        self.idx2label = data_dict['idx2label']
        self.label2idx = data_dict['label2idx']
        self.num_label = max(self.idx2label.keys()) + 1

        assert split in data_dict
        # assert split == 'test'
        self.split = split
        self.data_dict = data_dict[split]
        self.table_dict = table_dict
        self.data_list = []

        self.cls_token = tokenizer.cls_token_id
        self.sep_token = tokenizer.sep_token_id

        self.max_cell_len = args.max_cell_len
        self.max_col = args.max_col - 1
        self.num_row_per_sample = args.num_row_per_sample

    def generate_epoch(self):

        num_row_per_sample = self.num_row_per_sample
        max_col = self.max_col

        self.data_list = []

        for tbname, data in self.data_dict.items():

            table = self.table_dict[tbname]

            if len(data['label']) == 0:
                continue

            if len(data['col_idx']) == 0:
                continue

            if len(data['col_idx']) <= max_col:
                col_idx = sorted([ int(idx[1]) for idx in data['col_idx'] ])
                col_idxs = copy.deepcopy([col_idx])
            else:
                col_idx_before = sorted([int(idx[1]) for i, idx in enumerate(data['col_idx']) if i < max_col ])
                col_idx_after = sorted([int(idx[1]) for i, idx in enumerate(data['col_idx']) if i >= max_col])
                col_idxs = copy.deepcopy([col_idx_before, col_idx_after])

            for cut, col_idx in enumerate(col_idxs):

                assert len(col_idx) <= max_col

                row_cnt = len(table)
                col_cnt = len(table[0])

                col_flag = True if col_cnt <= max_col else False
                tb_row_nos = np.arange(0, row_cnt).tolist()

                for i in range( len(tb_row_nos) // num_row_per_sample ):

                    if col_flag:
                        col_nos = np.arange(0, col_cnt).tolist()
                        cls_idx = col_idx
                    else:
                        #add the first cls token for subject
                        population = np.setdiff1d(np.arange(1, col_cnt), col_idx)
                        col_nos = np.sort(
                            np.append([0],
                                      np.append(col_idx,
                                                np.random.choice(population, size=max_col-len(col_idx), replace=False)
                                                )
                                      )
                        )
                        cls_idx = np.where(np.isin(col_nos, np.array(col_idx)))[0].tolist()
                        col_nos = col_nos.tolist()

                    row_nos = tb_row_nos[ i*num_row_per_sample : (i+1)*num_row_per_sample ]

                    assert len(row_nos) == num_row_per_sample
                    assert len(cls_idx) == len(col_idx)

                    self.data_list.append(copy.deepcopy([tbname, row_nos, col_nos, cls_idx, col_idx, cut]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        item = {}

        tbname, row_nos, col_nos, cls_idx, col_idx, cut = self.data_list[idx]

        item['tbname'] = tbname
        item['col_idx'] = col_idx

        input_ids = []
        table = np.array(self.table_dict[tbname], dtype=object)
        for col in [ col for col in table[row_nos, :][:, col_nos].T ][::-1]:
            # bert ele: [CLS] text [SEP] => ele[1:-1]: text
            # qwen ele: text '<|endoftext|>
            for ele in col:
                if len(ele[:-1]) <= self.max_cell_len:
                    input_ids.extend(ele)
                else:
                    input_ids.extend(ele[:self.max_cell_len] + [ tokenizer.pad_token_id ] )
        assert len(input_ids) <= seq_len_limit

        item['input_ids'] = torch.tensor(input_ids, dtype=torch.int)
        item['attention_mask'] = torch.tensor([1] * len(input_ids), dtype=torch.int)
        item['cls_idx'] = cls_idx
        item['col_nos'] = col_nos
        item['tbname'] = tbname
        item['cut'] = cut

        return item


class CPADataset(Dataset):

    def __init__(self,
                 data_dict: dict,
                 table_dict: dict,
                 split: str,
                 args):

        self.idx2label = data_dict['idx2label']
        self.label2idx = data_dict['label2idx']
        self.num_label = max(self.idx2label.keys()) + 1

        assert split in data_dict
        self.split = split
        self.data_dict = data_dict[split]
        self.table_dict = table_dict
        self.data_list = []

        self.cls_token = tokenizer.cls_token_id
        self.sep_token = tokenizer.sep_token_id

        self.max_cell_len = args.max_cell_len
        self.max_col = args.max_col - 1
        self.max_row = args.max_row
        self.num_row_per_sample = args.num_row_per_sample

    def generate_epoch(self):

        max_col = self.max_col
        max_row = self.max_row
        num_row_per_sample = self.num_row_per_sample

        self.data_list = []

        for tbname, data in self.data_dict.items():

            if len(data['label']) == 0:
                continue

            table = self.table_dict[tbname]

            combos = [(l, int(idx[1])) for l, idx in zip(data['label'], data['col_idx'])]

            if len(data['col_idx']) <= max_col:
                chose_combos = sorted(combos, key=lambda x: x[1])
            else:
                chose_combos = sorted(random.sample(combos, max_col), key=lambda x: x[1])

            label = []
            col_idx = []
            for (l, idx) in chose_combos:
                label.append(l)
                col_idx.append(idx)

            row_cnt = len(table)
            col_cnt = len(table[0])

            col_flag = True if col_cnt <= max_col else False
            row_flag = True if row_cnt <= max_row else False

            if self.split == 'train':
                if row_flag:
                    tb_row_nos = np.random.choice(np.arange(0,row_cnt), size=row_cnt, replace=False).tolist()
                else:
                    tb_row_nos = np.random.choice(np.arange(0,row_cnt), size=max_row, replace=False).tolist()
                assert len(tb_row_nos) // num_row_per_sample <= max_row / num_row_per_sample
            else:
                tb_row_nos = np.arange(0, row_cnt).tolist()

            for i in range( len(tb_row_nos) // num_row_per_sample ):

                if col_flag:
                    col_nos = np.arange(0, col_cnt).tolist()
                    cls_idx = col_idx
                else:
                    population = np.setdiff1d(np.arange(1, col_cnt), col_idx)
                    col_nos = np.sort(
                        np.append([0],
                                  np.append(col_idx,
                                            np.random.choice(population, size=max_col-len(col_idx), replace=False)
                                            )
                                  )
                    )
                    cls_idx = np.where(np.isin(col_nos, np.array(col_idx)))[0].tolist()
                    col_nos = col_nos.tolist()

                row_nos = tb_row_nos[ i*num_row_per_sample : (i+1)*num_row_per_sample ]

                assert len(row_nos) == num_row_per_sample
                assert len(cls_idx) == len(label)

                self.data_list.append(copy.deepcopy([tbname, row_nos, col_nos, cls_idx, label]))

                if single_subtable_on and self.split == "train":
                    break

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        item = {}

        tbname, row_nos, col_nos, cls_idx, label = self.data_list[idx]

        input_ids = []
        table = np.array(self.table_dict[tbname], dtype=object)
        # reversed
        for col in [ col for col in table[row_nos, :][:, col_nos].T ][::-1]:
            # bert ele: [CLS] text [SEP] => ele[1:-1]: text
            # qwen ele: text '<|endoftext|>
            for ele in col:
                if len(ele[:-1]) <= self.max_cell_len:
                    input_ids.extend(ele)
                else:
                    input_ids.extend(ele[:self.max_cell_len] + [ tokenizer.pad_token_id ] )
        assert len(input_ids) <= seq_len_limit

        item['input_ids'] = torch.tensor(input_ids, dtype=torch.int)
        item['attention_mask'] = torch.tensor([1] * len(input_ids), dtype=torch.int)

        item['cls_idx'] = cls_idx
        item['label'] = label
        item['col_nos'] = col_nos
        item['tbname'] = tbname

        return item

def test_cta(split, data_dict, table_dict):
    start_time = time.time()

    if split != "test":
        train_cta_sch_dataset = CTADataset(data_dict, table_dict, split, args)
        train_cta_sch_dataset.generate_epoch()

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")

        batch_size = 128
        print('-*-', len(train_cta_sch_dataset))
        train_sampler = RandomSampler(train_cta_sch_dataset)
        train_dataloader = DataLoader(train_cta_sch_dataset,
                                      sampler=train_sampler,
                                      batch_size=batch_size,
                                      collate_fn=collate_fn)

        for epoch in range(1):
            train_cta_sch_dataset.generate_epoch()
            for batch_idx, batch in enumerate(train_dataloader):
                print("batch idx", batch_idx)
                print(batch)
    else:
        test_cta_sch_dataset = CTATestDataset(data_dict, table_dict, split, args)
        test_cta_sch_dataset.generate_epoch()

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")

        batch_size = 128
        test_dataloader = DataLoader(test_cta_sch_dataset,
                                      batch_size=batch_size,
                                      collate_fn=collate_test_fn)

        test_cta_sch_dataset.generate_epoch()
        for batch_idx, batch in enumerate(test_dataloader):
            print("batch idx", batch_idx)
            print(batch)

    mid_time = time.time()
    execution_time = mid_time - start_time
    print("Execution time:", execution_time, "seconds")

def test_cpa(split, data_dict, table_dict):

    start_time = time.time()

    if split != "test":
        train_cpa_sch_dataset = CPADataset(data_dict, table_dict, split, args)
        train_cpa_sch_dataset.generate_epoch()

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")

        batch_size = 128
        train_sampler = RandomSampler(train_cpa_sch_dataset)
        train_dataloader = DataLoader(train_cpa_sch_dataset,
                                      sampler=train_sampler,
                                      batch_size=batch_size,
                                      collate_fn=collate_fn)

        for epoch in range(1):
            train_cpa_sch_dataset.generate_epoch()
            for batch_idx, batch in enumerate(train_dataloader):
                print("batch idx", batch_idx)
                print(batch)
    else:
        test_cpa_sch_dataset = CPATestDataset(data_dict, table_dict, split, args)
        test_cpa_sch_dataset.generate_epoch()

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")

        batch_size = 128
        test_dataloader = DataLoader(test_cpa_sch_dataset,
                                      batch_size=batch_size,
                                      collate_fn=collate_test_fn)

        test_cpa_sch_dataset.generate_epoch()
        for batch_idx, batch in enumerate(test_dataloader):
            print("batch idx", batch_idx)
            print(batch)

    mid_time = time.time()
    execution_time = mid_time - start_time
    print("Execution time:", execution_time, "seconds")

if __name__ == "__main__":

    # testing this script:
    # reset the global variable for using the right tokenizer:
    # shortcut_name = args.shortcut_name => shortcut_name = "Qwen/Qwen3-Embedding-8B"
    dataset_path = "/apollo/users/dya/dataset/semtab"
    filename1 = os.path.join(dataset_path, 'CTA-SCH_llm.pkl')
    filename2 = os.path.join(dataset_path, 'cta_llm_qwen.pkl')
    data_dict = load(filename1)
    table_dict = load(filename2)

    test_cta('train', data_dict, table_dict)
    test_cta('validation', data_dict, table_dict)
    test_cta('test', data_dict, table_dict)