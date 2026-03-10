import os
import glob
import pickle
import json
import copy
from transformers import AutoTokenizer

from utils.file_registry import get_path

# downgrade numpy
# pip install numpy==1.20.3

shortcut_name = 'bert-base-multilingual-cased'
# shortcut_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(shortcut_name)

dataset_path = "/apollo/users/dya/dataset/semtab"

# data source : CTA_TABLE -> the folder with cta json files ; CPA_TABLE -> the folder with cpa json files
# there are different tables and overlapping tables between CTA_TABLE and CPA_TABLE
# CTA_TABLE and CPA_TABLE are placeholders

CTA_DATASET = os.path.join(dataset_path, 'CTA_TABLE')
CPA_DATASET = os.path.join(dataset_path, 'CPA_TABLE')

CTA_PICKLE = os.path.join(dataset_path, 'cta.pkl')
CPA_PICKLE = os.path.join(dataset_path, 'cpa.pkl')

SCH_CTA_PKL = os.path.join(dataset_path, "CTA-SCH.pkl")
SCH_CPA_PKL = os.path.join(dataset_path, "CPA-SCH.pkl")
DBP_CTA_PKL = os.path.join(dataset_path, "CTA-DBP.pkl")
DBP_CPA_PKL = os.path.join(dataset_path, "CPA-DBP.pkl")

TRAIN_SCH_CTA = get_path(dataset_path, 'CTA_SCH_TRAIN_GT_CSV', 'sotab')
TRAIN_SCH_CPA = get_path(dataset_path, 'CPA_SCH_TRAIN_GT_CSV', 'sotab')
TRAIN_DBP_CTA = get_path(dataset_path, 'CTA_DBP_TRAIN_GT_CSV', 'sotab')
TRAIN_DBP_CPA = get_path(dataset_path, 'CPA_DBP_TRAIN_GT_CSV', 'sotab')

VALID_SCH_CTA = get_path(dataset_path, 'CTA_SCH_VAL_GT_CSV', 'sotab')
VALID_SCH_CPA = get_path(dataset_path, 'CPA_SCH_VAL_GT_CSV', 'sotab')
VALID_DBP_CTA = get_path(dataset_path, 'CTA_DBP_VAL_GT_CSV', 'sotab')
VALID_DBP_CPA = get_path(dataset_path, 'CPA_DBP_VAL_GT_CSV', 'sotab')

TEST_SCH_CTA = get_path(dataset_path, 'CTA_SCH_TEST_GT_CSV', 'sotab')
TEST_SCH_CPA = get_path(dataset_path, 'CPA_SCH_TEST_GT_CSV', 'sotab')
TEST_DBP_CTA = get_path(dataset_path, 'CTA_DBP_TEST_GT_CSV', 'sotab')
TEST_DBP_CPA = get_path(dataset_path, 'CPA_DBP_TEST_GT_CSV', 'sotab')

CTA_FILES = [ f for f in glob.glob(os.path.join(CTA_DATASET, '*.json'))]
CPA_FILES = [ f for f in glob.glob(os.path.join(CPA_DATASET, '*.json'))]

cta_dbp_label_fn = get_path(dataset_path, 'CTA_DBP_LABEL_TXT', 'sotab')
cpa_dbp_label_fn= get_path(dataset_path, 'CPA_DBP_LABEL_TXT', 'sotab')
cta_sch_label_fn = get_path(dataset_path, 'CTA_SCH_LABEL_TXT', 'sotab')
cpa_sch_label_fn = get_path(dataset_path, 'CPA_SCH_LABEL_TXT', 'sotab')

label2idx = {
    'CTA-DBP': { line.strip() : idx for idx, line in enumerate(open(cta_dbp_label_fn,'r').readlines()) },
    'CPA-DBP': { line.strip() : idx for idx, line in enumerate(open(cpa_dbp_label_fn,'r').readlines()) },
    'CTA-SCH': { line.strip(): idx for idx, line in enumerate(open(cta_sch_label_fn, 'r').readlines())},
    'CPA-SCH': { line.strip(): idx for idx, line in enumerate(open(cpa_sch_label_fn, 'r').readlines())},
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

def save_cta_data_semtab(data, fn, src):
    pkls = {}
    for mode, dst in zip(['train', 'validation', 'test'], data):
        pkl = {}
        with open(dst,'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue

                splits = line.strip().split(',')
                if mode != "test":
                    fname, col, type = splits[0][:-3], splits[1], splits[2]
                else:
                    fname, col = splits[0][:-3], splits[1]

                if fname not in pkl:
                    if mode != "test":
                        pkl[fname] = {'col_idx':[], 'label':[]}
                    else:
                        pkl[fname] = {'col_idx':[]}

                pkl[fname]['col_idx'].append(col)
                if mode != "test":
                    type = label2idx[src][type]
                    pkl[fname]['label'].append(type)
        pkls[mode] = copy.deepcopy(pkl)
        assert id(pkls[mode]) != pkl

    pkls['label2idx'] = label2idx[src]
    pkls['idx2label'] = idx2label[src]
    save(fn, pkls)

def save_cpa_data_semtab(data, fn, src):
    pkls = {}
    for mode, dst in zip(['train', 'validation', 'test'], data):
        pkl = {}
        with open(dst, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue

                splits = line.strip().split(',')
                if mode != "test":
                    fname, s, o, p = splits[0][:-3], splits[1], splits[2], splits[3]
                else:
                    fname, s, o = splits[0][:-3], splits[1], splits[2]

                if fname not in pkl:
                    if mode != "test":
                        pkl[fname] = {'col_idx': [], 'label': []}
                    else:
                        pkl[fname] = {'col_idx': []}

                pkl[fname]['col_idx'].append( (s,o) )
                if mode != "test":
                    p = label2idx[src][p]
                    pkl[fname]['label'].append(p)
        pkls[mode] = copy.deepcopy(pkl)
        assert id(pkls[mode]) != pkl

    pkls['label2idx'] = label2idx[src]
    pkls['idx2label'] = idx2label[src]
    save(fn, pkls)

def save_cta_data(data, fn, src):
    pkls = {}
    for mode, dst in zip(['train', 'validation', 'test'], data):
        pkl = {}
        with open(dst,'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue

                splits = line.strip().split(',')
                fname, col, type = splits[0][:-3], splits[1], splits[2]

                if fname not in pkl:
                    pkl[fname] = {'col_idx': [], 'label': []}

                pkl[fname]['col_idx'].append(col)
                type = label2idx[src][type]
                pkl[fname]['label'].append(type)

        pkls[mode] = copy.deepcopy(pkl)
        assert id(pkls[mode]) != pkl

    pkls['label2idx'] = label2idx[src]
    pkls['idx2label'] = idx2label[src]
    save(fn, pkls)

def save_cpa_data(data, fn, src):
    pkls = {}
    for mode, dst in zip(['train', 'validation', 'test'], data):
        pkl = {}
        with open(dst, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue

                splits = line.strip().split(',')
                fname, s, o, p = splits[0][:-3], splits[1], splits[2], splits[3]

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
        pkl[tname_abbr] = []
        with open(tname, 'r') as f:
            for line in f.readlines():
                cell_values = []
                _dict = json.loads(line)
                for key, value in _dict.items():
                    if value:
                        cell_values.append(str(value))
                    else:
                        cell_values.append("null")

                cell_lm_indices = tokenizer(cell_values, padding=False, truncation=False, max_length=None, is_split_into_words=False,
                          return_tensors='np', return_length=True)['input_ids'].tolist()
                assert len(cell_lm_indices) == int(key) + 1
                pkl[tname_abbr].append(copy.deepcopy(cell_lm_indices))
    assert len(pkl) == len(tables)
    save(fn, pkl)

def compare_dict_values(dict1, dict2):
    # Find the common keys
    common_keys = set(dict1.keys()).intersection(dict2.keys())

    # Compare values for each common key
    for key in common_keys:
        if dict1[key] != dict2[key]:
            return False
    return True

def merge_dicts(dict1, dict2):
    merged_dict = {}

    # Find common keys
    common_keys = set(dict1.keys()).intersection(dict2.keys())

    # Add keys from dict1 and dict2 to the merged dictionary
    for key in dict1:
        if key in common_keys:
            if dict1[key] == dict2[key]:
                merged_dict[key] = dict1[key]
        else:
            merged_dict[key] = dict1[key]

    for key in dict2:
        if key not in merged_dict:
            merged_dict[key] = dict2[key]

    return merged_dict

if __name__ == "__main__":

    # generate two pickle files for CTA & CPA (tables)
    # dictionary
    # key: filename
    # values: 2d list

    save_tables(CTA_FILES, CTA_PICKLE)
    save_tables(CPA_FILES, CPA_PICKLE)

    # annotation files

    save_cta_data([TRAIN_SCH_CTA, VALID_SCH_CTA, TEST_SCH_CTA], SCH_CTA_PKL, 'CTA-SCH')
    save_cpa_data([TRAIN_SCH_CPA, VALID_SCH_CPA, TEST_SCH_CPA], SCH_CPA_PKL, 'CPA-SCH')

    save_cta_data([TRAIN_DBP_CTA, VALID_DBP_CTA, TEST_DBP_CTA], DBP_CTA_PKL, 'CTA-DBP')
    save_cpa_data([TRAIN_DBP_CPA, VALID_DBP_CPA, TEST_DBP_CPA], DBP_CPA_PKL, 'CPA-DBP')
