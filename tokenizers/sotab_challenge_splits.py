import os
import glob
import pickle
import json
import copy

from utils.file_registry import get_path

# downgrade numpy
# pip install numpy==1.20.3

# SAME DATASOURCE
# ONLY ANNOTATIONs
# creating special test splits (subsets of the full test split)

dataset_path = "/apollo/users/dya/dataset/semtab"

SCH_CTA_PKL_corner_cases = os.path.join(dataset_path, "CTA-SCH_test_corner_cases.pkl")
SCH_CTA_PKL_format_heterogeneity = os.path.join(dataset_path, "CTA-SCH_test_format_heterogeneity.pkl")
SCH_CTA_PKL_missing_values = os.path.join(dataset_path, "CTA-SCH_test_missing_values.pkl")
SCH_CTA_PKL_random = os.path.join(dataset_path, "CTA-SCH_test_random.pkl")

SCH_CPA_PKL_corner_cases = os.path.join(dataset_path, "CPA-SCH_test_corner_cases.pkl")
SCH_CPA_PKL_format_heterogeneity = os.path.join(dataset_path, "CPA-SCH_test_format_heterogeneity.pkl")
SCH_CPA_PKL_missing_values = os.path.join(dataset_path, "CPA-SCH_test_missing_values.pkl")
SCH_CPA_PKL_random = os.path.join(dataset_path, "CPA-SCH_test_random.pkl")

TEST_SCH_CTA_corner_cases = os.path.join(dataset_path, 'sotab_v2_cta_corner_cases_test_set.csv')
TEST_SCH_CTA_format_heterogeneity = os.path.join(dataset_path, 'sotab_v2_cta_format_heterogeneity_test_set.csv')
TEST_SCH_CTA_missing_values = os.path.join(dataset_path, 'sotab_v2_cta_missing_values_test_set.csv')
TEST_SCH_CTA_random = os.path.join(dataset_path, 'sotab_v2_cta_random_test_set.csv')

TEST_SCH_CPA_corner_cases = os.path.join(dataset_path, 'sotab_v2_cpa_corner_cases_test_set.csv')
TEST_SCH_CPA_format_heterogeneity = os.path.join(dataset_path, 'sotab_v2_cpa_format_heterogeneity_test_set.csv')
TEST_SCH_CPA_missing_values = os.path.join(dataset_path, 'sotab_v2_cpa_missing_values_test_set.csv')
TEST_SCH_CPA_random = os.path.join(dataset_path, 'sotab_v2_cpa_random_test_set.csv')

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

def save_cta_data(data, fn, src):
    pkls = {}
    for mode, dst in zip(['test'], data):
        pkl = {}
        with open(dst,'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue

                splits = line.strip().split(',')
                fname, col, type = splits[0][:-3], splits[1], splits[2]

                if type not in label2idx[src]:
                    continue

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
    for mode, dst in zip(['test'], data):
        pkl = {}
        with open(dst, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue

                splits = line.strip().split(',')
                fname, s, o, p = splits[0][:-3], splits[1], splits[2], splits[3]

                if p not in label2idx[src]:
                    continue

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

if __name__ == "__main__":

    # special challenges
    # annotation files only

    save_cta_data([TEST_SCH_CTA_corner_cases], SCH_CTA_PKL_corner_cases, 'CTA-SCH')
    save_cta_data([TEST_SCH_CTA_format_heterogeneity], SCH_CTA_PKL_format_heterogeneity, 'CTA-SCH')
    save_cta_data([TEST_SCH_CTA_missing_values], SCH_CTA_PKL_missing_values, 'CTA-SCH')
    save_cta_data([TEST_SCH_CTA_random], SCH_CTA_PKL_random, 'CTA-SCH')

    save_cpa_data([TEST_SCH_CPA_corner_cases], SCH_CPA_PKL_corner_cases, 'CPA-SCH')
    save_cpa_data([TEST_SCH_CPA_format_heterogeneity], SCH_CPA_PKL_format_heterogeneity, 'CPA-SCH')
    save_cpa_data([TEST_SCH_CPA_missing_values], SCH_CPA_PKL_missing_values, 'CPA-SCH')
    save_cpa_data([TEST_SCH_CPA_random], SCH_CPA_PKL_random, 'CPA-SCH')