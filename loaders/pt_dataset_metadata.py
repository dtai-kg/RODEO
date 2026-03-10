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
from utils.file_registry import get_path

args = load_args_from_parser()
# args.max_col = 20
# args.max_cell_len = 8
# args.num_row_per_sample = 2
single_subtable_on = args.single_subtable_on
shortcut_name = args.shortcut_name
tokenizer = AutoTokenizer.from_pretrained(shortcut_name)

print(args)

# print(tokenizer.cls_token_id)
# print(tokenizer.sep_token_id)
# print(tokenizer.pad_token_id)
# print(tokenizer.mask_token_id)

table_to_cluster_path = get_path(args.gittab_dataset_path, "TABLE_TO_CLUSTER_PKL_FILE_PATH", dataset='gittab')
with open(table_to_cluster_path, "rb") as f:
    table_to_cluster = pickle.load(f)

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

"""
Example:
input_ids:  [[0, 1], [0, 5], [0, 3], [1, 2], [1, 3]]
cls_idx: [[1,2], [1]]
output: [[0, 5], [0, 3], [1, 3]]
"""
def retrieve(input_ids, cls_idx, tbnames, col_nos):

    first_filter = torch.nonzero(input_ids == tokenizer.cls_token_id)
    # we need to filter out unannotated (context) columns
    output = []
    for sample_idx, token_idx in enumerate(cls_idx):
        mask = (first_filter[:, 0] == sample_idx)
        # print(tbnames[sample_idx])
        # print("token idx:",token_idx)
        # print("cls token in total:",len(first_filter[mask]))
        # print("col_nos:", col_nos[sample_idx])
        # print("label:", label)
        rows = first_filter[mask][token_idx]
        output.append(rows)
    return torch.cat(output)

def collate_rel_fn(samples):

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [sample["input_ids"] for sample in samples], padding_value=tokenizer.pad_token_id
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [sample["attention_mask"] for sample in samples]
    )

    label = torch.cat([torch.tensor(sample["label"]) for sample in samples])

    tbnames = []
    topic = []
    for sample in samples:
        for _ in range(len(sample["label"])):
            tbnames.append(sample["tbname"])
            topic.append(table_to_cluster[sample["tbname"]])
    col_nos = [sample["col_nos"] for sample in samples]
    cls_idx = [sample["cls_idx"] for sample in samples]
    cls_idx = retrieve(input_ids.T, cls_idx, tbnames, col_nos)

    batch = {"input_ids": input_ids.T, "attention_mask": attention_mask.T, "cls_idx": cls_idx, "label": label, "topic": topic, "tbname": tbnames}
    return batch

def collate_test_fn(samples):

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [sample["input_ids"] for sample in samples], padding_value=tokenizer.pad_token_id)

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [sample["attention_mask"] for sample in samples])

    cls_idx = [sample["cls_idx"] for sample in samples]
    col_nos = [sample["col_nos"] for sample in samples]
    tbnames = [sample["tbname"] for sample in samples]
    # Added: cuts based on table columns for later inference
    cuts = [sample["cut"] for sample in samples]

    cls_idx = retrieve(input_ids.T, cls_idx, tbnames, col_nos)
    batch = {"input_ids": input_ids.T, "attention_mask": attention_mask.T, "cls_idx": cls_idx, "tbname": tbnames, "col_nos": col_nos, "cut": cuts}

    return batch

def collate_fn(samples):

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [sample["input_ids"] for sample in samples], padding_value=tokenizer.pad_token_id
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [sample["attention_mask"] for sample in samples]
    )

    label = torch.cat([torch.tensor(sample["label"]) for sample in samples])

    tbnames = []
    # topic = []
    for sample in samples:
        for _ in range(len(sample["label"])):
            tbnames.append(sample["tbname"])

    col_nos = [sample["col_nos"] for sample in samples]
    cls_idx = [sample["cls_idx"] for sample in samples]
    cls_idx = retrieve(input_ids.T, cls_idx, tbnames, col_nos)
    batch = {"input_ids": input_ids.T, "attention_mask": attention_mask.T, "cls_idx": cls_idx, "label": label, "tbname": tbnames}

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

            table = self.table_dict[tbname]["cells"]

            if len(data['col_idx']) <= max_col:
                col_idx = sorted([ int(idx) for idx in data['col_idx'] ])
                col_idxs = copy.deepcopy([col_idx])
            else:
                # Split data['col_idx'] into chunks of size max_col
                col_idxs = [
                    sorted([int(idx) for idx in data['col_idx'][i:i + max_col]])
                    for i in range(0, len(data['col_idx']), max_col)
                ]
                col_idxs = copy.deepcopy(col_idxs)

                # Optional: check that all chunks except possibly the last have size <= max_col
                for chunk in col_idxs:
                    assert len(chunk) <= max_col

            assert len(data['col_idx']) == len(set(data['col_idx'])), print(tbname)

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

                        assert col_idx[0] > 0
                        population = np.setdiff1d(np.arange(0, col_cnt), [x - 1 for x in col_idx])
                        col_nos = np.sort(np.append([x - 1 for x in col_idx], np.random.choice(population, size=max_col - len(col_idx),  replace=False)))
                        cls_idx = np.where(np.isin(col_nos, np.array([x - 1 for x in col_idx])))[0].tolist()
                        cls_idx = [x + 1 for x in cls_idx] # title as the first column
                        assert cls_idx[0] > 0
                        col_nos = col_nos.tolist()
                        assert col_nos[0] >= 0

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

        table = np.array(self.table_dict[tbname]['cells'], dtype=object)
        col_names = np.array(self.table_dict[tbname]['col_names'], dtype=object)
        title = np.array(self.table_dict[tbname]['title'], dtype=object)[0]

        input_ids.append(self.cls_token)
        input_ids.extend(title[1:-1])

        for col_no, col in zip(col_nos, table[row_nos,:][:,col_nos].T):

            input_ids.append(self.cls_token)

            if len(col_names[1:-1]) <= self.max_cell_len:
                input_ids.extend(col_names[col_no][1:-1])
            else:
                input_ids.extend(col_names[col_no][1:self.max_cell_len + 1])

            for ele in col:
                if len(ele[1:-1]) <= self.max_cell_len:
                    input_ids.extend(ele[1:-1])
                else:
                    input_ids.extend(ele[1:self.max_cell_len+1])

        input_ids.append(self.sep_token)
        assert len(input_ids) <= 512

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

            table = self.table_dict[tbname]["cells"]

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
                    # we could use the orignal indices when the number of columns is smaller than the max number of columns (threshold)
                else:
                    if 0 in col_idx:
                        population = np.setdiff1d(np.arange(0, col_cnt), [x - 1 for x in col_idx if x != 0 ])
                        col_nos = np.sort( np.append([x - 1 for x in col_idx if x != 0 ], np.random.choice(population, size=max_col-len(col_idx) + 1, replace=False)))
                        cls_idx = np.where(np.isin(col_nos, np.array([x - 1 for x in col_idx if x != 0])))[0].tolist()
                        cls_idx = [0] + [x + 1 for x in cls_idx]
                        col_nos = col_nos.tolist()
                    else:
                        population = np.setdiff1d(np.arange(0, col_cnt), [x - 1 for x in col_idx])
                        col_nos = np.sort( np.append([x - 1 for x in col_idx], np.random.choice(population, size=max_col-len(col_idx), replace=False)))
                        cls_idx = np.where(np.isin(col_nos, np.array([x - 1 for x in col_idx])))[0].tolist()
                        cls_idx = [x + 1 for x in cls_idx]
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

        table = np.array(self.table_dict[tbname]['cells'], dtype=object)
        col_names = np.array(self.table_dict[tbname]['col_names'], dtype=object)
        title = np.array(self.table_dict[tbname]['title'], dtype=object)[0]

        input_ids.append(self.cls_token)
        input_ids.extend(title[1:-1])

        for col_no, col in zip(col_nos, table[row_nos,:][:,col_nos].T):

            input_ids.append(self.cls_token)
            if len(col_names[1:-1]) <= self.max_cell_len:
                input_ids.extend(col_names[col_no][1:-1])
                # print (col_names[col_no][1:-1])
            else:
                input_ids.extend(col_names[col_no][1:self.max_cell_len + 1])

            for ele in col:
                if len(ele[1:-1]) <= self.max_cell_len:
                    input_ids.extend(ele[1:-1])
                else:
                    input_ids.extend(ele[1:self.max_cell_len+1])

        input_ids.append(self.sep_token)
        assert len(input_ids) <= 512

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

            table = self.table_dict[tbname]["cells"]

            if len(data['label']) == 0:
                continue

            if len(data['col_idx']) == 0:
                continue

            if len(data['col_idx']) <= max_col:
                col_idx = sorted([ int(idx[1]) for idx in data['col_idx'] ])
                col_idxs = copy.deepcopy([col_idx])
            else:
                # Split data['col_idx'] into chunks of size max_col
                col_idxs = [
                    sorted([int(idx[1]) for idx in data['col_idx'][i:i + max_col]])
                    for i in range(0, len(data['col_idx']), max_col)
                ]
                col_idxs = copy.deepcopy(col_idxs)

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

                        population = np.setdiff1d(np.arange(0, col_cnt), [x - 1 for x in col_idx])
                        col_nos = np.sort(np.append([x - 1 for x in col_idx], np.random.choice(population, size=max_col - len(col_idx),  replace=False)))

                        cls_idx = np.where(np.isin(col_nos, np.array([x - 1 for x in col_idx])))[0].tolist()
                        cls_idx = [x + 1 for x in cls_idx] # title as the first column
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
        table = np.array(self.table_dict[tbname]['cells'], dtype=object)
        col_names = np.array(self.table_dict[tbname]['col_names'], dtype=object)
        title = np.array(self.table_dict[tbname]['title'], dtype=object)[0]

        input_ids.append(self.cls_token)
        input_ids.extend(title[1:-1])

        for col_no, col in zip(col_nos, table[row_nos,:][:,col_nos].T):

            input_ids.append(self.cls_token)
            if len(col_names[1:-1]) <= self.max_cell_len:
                input_ids.extend(col_names[col_no][1:-1])
            else:
                input_ids.extend(col_names[col_no][1:self.max_cell_len + 1])

            for ele in col:
                if len(ele[1:-1]) <= self.max_cell_len:
                    input_ids.extend(ele[1:-1])
                else:
                    input_ids.extend(ele[1:self.max_cell_len+1])

        input_ids.append(self.sep_token)
        assert len(input_ids) <= 512

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

            table = self.table_dict[tbname]["cells"]

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
                    population = np.setdiff1d(np.arange(0, col_cnt), [x - 1 for x in col_idx])
                    col_nos = np.sort(
                        # np.append([0],
                                  np.append( [x - 1 for x in col_idx],
                                            np.random.choice(population, size=max_col-len(col_idx), replace=False)
                                            )
                        #          )
                    )
                    cls_idx = np.where(np.isin(col_nos, np.array([x - 1 for x in col_idx])))[0].tolist()
                    cls_idx = [x + 1 for x in cls_idx]  # title as the first column
                    col_nos = col_nos.tolist()

                row_nos = tb_row_nos[ i*num_row_per_sample : (i+1)*num_row_per_sample ]

                assert len(row_nos) == num_row_per_sample
                assert len(cls_idx) == len(label), print (len(cls_idx), len(label))

                self.data_list.append(copy.deepcopy([tbname, row_nos, col_nos, cls_idx, label]))

                if single_subtable_on and self.split == "train":
                    break

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        item = {}

        tbname, row_nos, col_nos, cls_idx, label = self.data_list[idx]

        input_ids = []
        table = np.array(self.table_dict[tbname]['cells'], dtype=object)
        col_names = np.array(self.table_dict[tbname]['col_names'], dtype=object)
        title = np.array(self.table_dict[tbname]['title'], dtype=object)

        input_ids.append(self.cls_token)
        input_ids.extend(title[1:-1])

        for col_no, col in zip(col_nos, table[row_nos,:][:,col_nos].T):

            input_ids.append(self.cls_token)
            if len(col_names[1:-1]) <= self.max_cell_len:
                input_ids.extend(col_names[col_no][1:-1])
            else:
                input_ids.extend(col_names[col_no][1:self.max_cell_len + 1])

            for ele in col:
                if len(ele[1:-1]) <= self.max_cell_len:
                    input_ids.extend(ele[1:-1])
                else:
                    input_ids.extend(ele[1:self.max_cell_len+1])

        input_ids.append(self.sep_token)
        assert len(input_ids) <= 512

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
        train_sampler = RandomSampler(train_cta_sch_dataset)
        train_dataloader = DataLoader(train_cta_sch_dataset,
                                      sampler=train_sampler,
                                      batch_size=batch_size,
                                      collate_fn=collate_fn)

        for epoch in range(1):
            train_cta_sch_dataset.generate_epoch()
            for batch_idx, batch in enumerate(train_dataloader):
                print("batch idx", batch_idx)
                # print(batch)
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
            # print(batch)

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
                # print(batch)
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
            # print(batch)

    mid_time = time.time()
    execution_time = mid_time - start_time
    print("Execution time:", execution_time, "seconds")

if __name__ == "__main__":

    # testing this script:
    # reset the global variables in case it reaches input length limits:
    # args.max_col = 20
    # args.max_cell_len = 8
    # args.num_row_per_sample = 2
    dataset_path = "/apollo/users/dya/dataset/gittable_numeric"

    filename1 = os.path.join(dataset_path, 'CTA-GIT.pkl')
    filename2 = os.path.join(dataset_path, 'git.pkl')
    data_dict = load(filename1)
    table_dict = load(filename2)

    test_cta('train', data_dict, table_dict)
    test_cta('dev', data_dict, table_dict)
    test_cta('test', data_dict, table_dict)

    filename1 = os.path.join(dataset_path, 'CPA-GIT.pkl')
    filename2 = os.path.join(dataset_path, 'git.pkl')
    data_dict = load(filename1)
    table_dict = load(filename2)

    test_cpa('train', data_dict, table_dict)
    test_cpa('dev', data_dict, table_dict)
    test_cpa('test', data_dict, table_dict)