import os
import glob
import pickle
import json
import copy
import pprint
from transformers import AutoTokenizer

import pandas as pd
import pyarrow.parquet as pq

from utils.file_registry import get_path
# downgrade numpy
# pip install numpy==1.20.3

shortcut_name = 'bert-base-multilingual-cased'
# shortcut_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(shortcut_name)

dataset_path = "/apollo/users/dya/dataset/gittable_numeric"

GIT_CTA_PKL = os.path.join(dataset_path, "CTA-GIT.pkl")
GIT_CPA_PKL = os.path.join(dataset_path, "CPA-GIT.pkl")

TRAIN_GIT_CTA = get_path(dataset_path, 'CTA_GIT_TRAIN_GT_CSV', 'gittab')
TRAIN_GIT_CPA = get_path(dataset_path, 'CPA_GIT_TRAIN_GT_CSV', 'gittab')

VALID_GIT_CTA = get_path(dataset_path, 'CTA_GIT_VAL_GT_CSV', 'gittab')
VALID_GIT_CPA = get_path(dataset_path, 'CPA_GIT_VAL_GT_CSV', 'gittab')

TEST_GIT_CTA = get_path(dataset_path, 'CTA_GIT_TEST_GT_CSV', 'gittab')
TEST_GIT_CPA = get_path(dataset_path, 'CPA_GIT_TEST_GT_CSV', 'gittab')

GIT_PICKLE = os.path.join(dataset_path, "git.pkl")

GIT_DATASET = get_path(dataset_path, 'RAW_TABLES_PATH', 'gittab')

cta_git_label_fn = get_path(dataset_path, 'CTA_GIT_LABEL_TXT', 'gittab')
cpa_git_label_fn = get_path(dataset_path, 'SYNTHETIC_REL_LABEL_TXT', 'gittab')

label2idx = {
    'CTA-GIT': { line.strip() : idx for idx, line in enumerate(open(cta_git_label_fn, 'r').readlines()) },
    'CPA-GIT': { line.strip() : idx for idx, line in enumerate(open(cpa_git_label_fn, 'r').readlines()) },
}

print(label2idx)

idx2label = {}

for task, map in label2idx.items():
    idx2label[task] = {}
    for label, idx in map.items():
        idx2label[task][idx] = label

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def save_cta_data(data, fn, src):
    pkls = {}
    for mode, dst in zip(['train', 'dev', 'test'], data):
        pkl = {}
        with open(dst,'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue

                splits = line.strip().split(',')
                fname, col, type, category = splits[0], splits[1], splits[2], splits[3]

                if fname not in pkl:
                    pkl[fname] = {'col_idx': [], 'label': [], 'category': []}

                type = label2idx[src][type]
                pkl[fname]['label'].append(type)
                pkl[fname]['col_idx'].append(col)
                pkl[fname]['category'].append(category)

        pkls[mode] = copy.deepcopy(pkl)
        assert id(pkls[mode]) != pkl

    pkls['label2idx'] = label2idx[src]
    pkls['idx2label'] = idx2label[src]
    save(fn, pkls)

def save_cpa_data(data, fn, src):
    pkls = {}
    for mode, dst in zip(['train', 'dev', 'test'], data):
        pkl = {}
        with open(dst, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue

                splits = line.strip().split(',')
                fname, s, o, p = splits[0], splits[1], splits[2], splits[3]

                if fname not in pkl:
                    pkl[fname] = {'col_idx': [], 'label': []}

                pkl[fname]['col_idx'].append( (s,o) )
                p = label2idx[src][p]
                pkl[fname]['label'].append(p)

        pkls[mode] = copy.deepcopy(pkl)
        assert id(pkls[mode]) != pkl

    pkls['label2idx'] = label2idx[src]
    pkls['idx2label'] = idx2label[src]
    save(fn, pkls)

def save_tables(tables, fn):
    pkl = {}

    for tname in tables:
        tname_abbr = tname.split('/')[-1]
        print(tname_abbr, flush=True)

        pkl[tname_abbr] = {'cells': [], 'col_names': None, 'title': None}
        # table = pq.read_table(tname)
        table = pq.ParquetFile(tname).read()

        # table name
        tname_lm_indices = tokenizer(tname, padding=False, truncation=False, max_length=None, is_split_into_words=False,
                    return_tensors='np', return_length=True)['input_ids'].tolist()
        pkl[tname_abbr]['title'] = copy.deepcopy(convert(tname_lm_indices))

        column_names = table.column_names
        col_names_lm_indices = tokenizer(column_names, padding=False, truncation=False, max_length=None, is_split_into_words=False,
                    return_tensors='np', return_length=True)['input_ids'].tolist()
        pkl[tname_abbr]['col_names'] = copy.deepcopy(convert(col_names_lm_indices))

        # cell content
        for row in table.to_batches()[0].to_pylist():
            cell_values = []
            for value in row.values():
                if value:
                    cell_values.append(str(value))
                else:
                    cell_values.append("null")

            cell_lm_indices = tokenizer(cell_values, padding=False, truncation=False, max_length=None, is_split_into_words=False,
                      return_tensors='np', return_length=True)['input_ids'].tolist()

            assert len(cell_lm_indices) == len(row.values())
            assert len(cell_lm_indices) == len(col_names_lm_indices)

            if len(cell_lm_indices) != len(col_names_lm_indices):
                print(tname_abbr, flush=True)
                print (tname_abbr, column_names, cell_values)
            pkl[tname_abbr]['cells'].append(copy.deepcopy(convert(cell_lm_indices)))

    save(fn, pkl)

def convert(batch_indices):
    data = []
    for indices in batch_indices:
        if not isinstance(indices, list):
            data.append(indices.tolist())
        else:
            data.append(indices)
    return data

if __name__ == "__main__":

    # generate one pickle file for both CTA & CPA (tables)
    # dictionary
    # key: filename
    # values: 2d list
    train_tables = set(pd.read_csv(TRAIN_GIT_CTA)["table_name"].dropna().astype(str).tolist())
    valid_tables = set(pd.read_csv(VALID_GIT_CTA)["table_name"].dropna().astype(str).tolist())
    test_tables = set(pd.read_csv(TEST_GIT_CTA)["table_name"].dropna().astype(str).tolist())
    target_tables = train_tables | valid_tables | test_tables
    GIT_TABLES = [ os.path.join(GIT_DATASET, tbname) for tbname in target_tables ]

    save_tables(GIT_TABLES, GIT_PICKLE)

    # annotation files
    save_cta_data([TRAIN_GIT_CTA, VALID_GIT_CTA, TEST_GIT_CTA], GIT_CTA_PKL, 'CTA-GIT')
    save_cpa_data([TRAIN_GIT_CPA, VALID_GIT_CPA, TEST_GIT_CPA], GIT_CPA_PKL, 'CPA-GIT')


