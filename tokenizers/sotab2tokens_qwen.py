import os
import glob
import pickle
import json
import copy
import gc
from transformers import AutoTokenizer

from utils.file_registry import get_path

shortcut_name = 'Qwen/Qwen3-Embedding-8B'
# shortcut_name = 'Qwen/Qwen3-Embedding-4B'
# shortcut_name = 'Qwen/Qwen3-Embedding-0.6B'
tokenizer = AutoTokenizer.from_pretrained(shortcut_name, padding_side='left')

dataset_path = "/apollo/users/dya/dataset/semtab"

# THE SAME SOTAB DATA SOURCE BUT WITH ONLY FEW TRAIN CPA ANNOTATIONS (small)
# CTA_TABLE and CPA_TABLE are placeholders
CTA_DATASET = os.path.join(dataset_path, 'CTA_TABLE')
CPA_DATASET = os.path.join(dataset_path, 'CPA_TABLE')

CTA_PICKLE = os.path.join(dataset_path, 'cta_full_qwen.pkl')
CPA_PICKLE = os.path.join(dataset_path, 'cpa_full_qwen.pkl')
CTA_PICKLE_LLM = os.path.join(dataset_path, 'cta_llm_qwen.pkl')
CPA_PICKLE_LLM = os.path.join(dataset_path, 'cpa_llm_qwen.pkl')

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

print(tokenizer.special_tokens_map)
# Print special token strings
print("Special tokens:")
print(tokenizer.special_tokens_map)

# Print special token IDs
print("\nSpecial token IDs:")
for name, token in tokenizer.special_tokens_map.items():
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"{name}: '{token}' → ID: {token_id}")

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
    # ONLY RETRIEVE RELEVANT TABLES TO AVOID LARGE PKL FILES (FAST LOADING DURING TRAINING)
    task_data = load(os.path.join(dataset_path, 'cta_bert_llm.pkl'))
    # task_data = load(os.path.join(dataset_path, 'cpa_bert_llm.pkl'))
    annotated_tbnames = list([ key.strip() for key in task_data.keys()])

    pkl = {}
    for t_idx, tname in enumerate(tables):
        tname_abbr = tname.split('/')[-1].strip()
        # duplicate tables in different folders
        if tname_abbr in pkl:
            continue
        if tname_abbr not in annotated_tbnames:
            continue
        print(f"[{t_idx + 1}/{len(annotated_tbnames)}] Processing table: {tname}", flush=True)
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
                          return_tensors='np', return_length=True)['input_ids']
                assert len(cell_lm_indices) == int(key) + 1

                tmp = []
                for indices in cell_lm_indices:
                    if not isinstance(cell_lm_indices, list):
                        tmp.append(indices.tolist())
                    else:
                        tmp.append(indices)
                cell_lm_indices = tmp
                pkl[tname_abbr].append(copy.deepcopy(cell_lm_indices))
                
    assert len(pkl) == len(annotated_tbnames)
    save(fn, pkl)


if __name__ == "__main__":

    # generate two pickle files for CTA & CPA (tables)
    # dictionary
    # key: filename
    # values: 2d list

    # save_tables(CTA_FILES, CTA_PICKLE)
    # save_tables(CPA_FILES, CPA_PICKLE)

    # save_tables(CTA_FILES, CTA_PICKLE_LLM)
    save_tables(CPA_FILES, CPA_PICKLE_LLM)

    # annotation files

    # save_cta_data([TRAIN_SCH_CTA, VALID_SCH_CTA, TEST_SCH_CTA], SCH_CTA_PKL, 'CTA-SCH')
    # save_cpa_data([TRAIN_SCH_CPA, VALID_SCH_CPA, TEST_SCH_CPA], SCH_CPA_PKL, 'CPA-SCH')
    #
    # save_cta_data([TRAIN_DBP_CTA, VALID_DBP_CTA, TEST_DBP_CTA], DBP_CTA_PKL, 'CTA-DBP')
    # save_cpa_data([TRAIN_DBP_CPA, VALID_DBP_CPA, TEST_DBP_CPA], DBP_CPA_PKL, 'CPA-DBP')


