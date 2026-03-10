import ast
import json
import pickle
import pprint
import copy
import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_seqs
from transformers import AutoTokenizer
import pprint

from utils.file_registry import get_path

# downgrade numpy
# pip install numpy==1.20.3

# finding: not all the cells are included within each column
# finding: similar/same columns with same values
# finding: it seems one table(e.g. 14679792) was split into several subtables with (e.g. 14679792-2, 14679792-3, 14679792-6)

shortcut_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(shortcut_name)

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def extract_table_data(table_dict):
    textData = []
    for row in table_dict['tableData']:
        textData.append({ str(col_id) : cell['text'] for col_id, cell in enumerate(row)})
    return textData

def read_table(fn):
    table_dicts = dict()
    with open(fn, 'r') as fp:
        for line in fp.readlines():
            table_dict = json.loads(line)
            _id = table_dict['_id']
            subject_column = table_dict['subject_column']
            table_dicts[_id] = {
                'subject_column': subject_column,
                'table': extract_table_data(table_dict)
            }
    return table_dicts

def read_vocab(fn):
    id2label = dict()
    label2id = dict()
    with open(fn, 'r') as fp:
        for line in fp.readlines():
            if len(line.strip()) == 0:
                continue
            id, label = line.split('\t')[0], line.split('\t')[1]
            id = int(id.strip())
            label = label.strip()
            id2label[id] = label
            label2id[label] = id
    return id2label, label2id

def extract_from_raw_table(table_dicts, task_dict, label2id, task='cta'):

    task_annotation = dict()
    task_table = dict()

    for id, task_data in task_dict.items():

        if task == 'cpa':
            raw_data = table_dicts[id]

            # train
            if id in ['15708593-10','7436515-4','7395229-5','5434392-2',  '5724621-4', '825486-1', '825692-1',
                      '825692-3', '1258230-2', '1258230-4', '1258230-5', '27504997-1', '5056631-6', '15708593-11', '15708593-13', '15708593-16', '15708593-17', '15708593-18', '15708593-19',
                      '15708593-20', '15708593-8', '15708593-9', '893856-1', '893856-2', '893856-4', '893856-6', '893856-7',
                      '893897-2', '7429107-5', '7436515-2', '7436515-3', '7436515-5', '7436515-6', '1179901-2', '2010901-1', '11190694-1', '11190694-15', '11190694-16', '11190694-17', '11190694-18',
                      '11190694-2', '11190694-3', '11190694-4', '11190694-6', '896212-1', '896212-2', '896212-3', '896212-4', '896212-6', '896212-7', '25055540-1', '23332726-1', '5162563-1', '5162563-2',
                      '5162563-3', '5162563-4', '38473731-3', '38473731-4', '21478832-3', '1164323-6', '1164323-7', '7395229-2', '7395229-3', '7395229-4', '7395229-6', '7403869-1', '7403869-2',
                      '7403869-3', '7403869-4', '7403869-5', '7403869-6', '7405057-2', '7405057-4', '7405057-6', '7405057-7', '7405281-2', '7405281-4', '7405281-5', '7405281-6', '7405281-7', '807662-1', '808619-7']:

                # original tables
                tmp_list = []
                for col_idx in range(len(raw_data['table'][0])):
                    tmp_list.append(' '.join([row[str(col_idx)] for row in raw_data['table']]))
                # find subject column
                col_text = ' '.join([cell[1][1] for cell in task_data['tableData'][0]])
                scores = normalized_damerau_levenshtein_distance_seqs(col_text, tmp_list)
                for tmp in np.argsort(scores).tolist():
                    if task_data['tableData'][0][0][1][1] ==\
                            raw_data['table'][ task_data['tableData'][0][0][0][0] ][str(tmp)]:
                        if tmp == 0:
                            subj_col_idx = tmp
                            print (raw_data['table'][0][str(tmp)])
                        else:
                            subj_col_idx = raw_data['subject_column']
                        break
                    else:
                        continue
            else:
                subj_col_idx = raw_data['subject_column']

            # modify original tables (subject column -> the first column)
            # tokenization_targets
            tmp_list_2 = [ ' '.join([row[str(subj_col_idx)] for row in raw_data['table']]) ]
            tmp_list_3 = [ [row[str(subj_col_idx)] for row in raw_data['table']] ]
            for col_idx in range(len(raw_data['table'][0])):
                if col_idx == subj_col_idx:
                    continue
                col_cell_values = [row[str(col_idx)] for row in raw_data['table']]
                col_text = ' '.join(col_cell_values)
                # remove empty column
                if len(col_text.strip()) == 0:
                    continue
                else:
                    tmp_list_2.append(col_text)
                    tmp_list_3.append(col_cell_values)

            # find the indexes of task columns from modified tables
            target_col_idx = []
            for col in task_data['tableData']:
                col_text = ' '.join([cell[1][1] for cell in col])
                scores = normalized_damerau_levenshtein_distance_seqs(col_text, tmp_list_2)
                target_col_idx.append(np.argmin(scores))

            target_col_idx_pair = []
            subj_idx = target_col_idx[0]
            for obj_idx in target_col_idx[1:]:
                target_col_idx_pair.append( (str(subj_idx), str(obj_idx)) )

            label = []
            for label_within_one_col in task_data['label']:
                label.append([ label2id[l] for l in label_within_one_col])

            task_annotation[id] = {'col_idx': target_col_idx_pair, 'label': label}

            task_table[id] = [
                {str(i): tmp_list_3[i][j] for i in range( len(tmp_list_3) ) } for j in range( len(tmp_list_3[0]) )
            ]

        else:
            # original tables
            tmp_list = []
            tmp_list_2 = []
            raw_data = table_dicts[id]
            for col_idx in range(len(raw_data['table'][0])):
                col_cell_values = [row[str(col_idx)] for row in raw_data['table']]
                col_text = ' '.join(col_cell_values)
                if len(col_text.strip()) == 0:
                    continue
                else:
                    tmp_list.append(col_text)
                    tmp_list_2.append(col_cell_values)

            # find the indexes of task columns from original tables
            target_col_idx = []
            for col_id, col in enumerate(task_data['tableData']):
                col_text = ' '.join([cell[1][1] for cell in col])
                scores = normalized_damerau_levenshtein_distance_seqs(col_text, tmp_list)
                for tmp in np.argsort(scores).tolist():
                    if col[0][1][1] != raw_data['table'][col[0][0][0]][str(tmp)]:
                        continue
                    else:
                        target_col_idx.append(tmp)
                        break

            label = []
            for label_within_one_col in task_data['label']:
                label.append([ label2id[l] for l in label_within_one_col])
            task_annotation[id] = {'col_idx': target_col_idx, 'label': label}

            task_table[id] = [
                {str(i): tmp_list_2[i][j] for i in range( len(tmp_list_2) ) } for j in range( len(tmp_list_2[0]) )
            ]

    return task_annotation, task_table

keys = ['_id', 'pgTitle', 'pgId', 'sectionTitle', 'tableCaption', 'processed_tableHeaders', 'tableData', 'label']

def read_cta(fn):

    with open(fn, 'r') as fp:
        content = fp.read()
        nested_list = ast.literal_eval(content)

    cta_id_list = [table_list[0] for table_list in nested_list]
    assert len(cta_id_list) == len(set(cta_id_list))

    cta_dict = {}
    for table_list in nested_list:
        cta_id = table_list[0]
        cta_dict[cta_id] = dict()
        assert len(keys) == len(table_list)
        for key, ele in zip(keys, table_list):
            cta_dict[cta_id][key] = ele

    return cta_dict

def read_cpa(fn):
    with open(fn, 'r') as fp:
        content = fp.read()
        nested_list = ast.literal_eval(content)

    cpa_id_list = [table_list[0] for table_list in nested_list]
    assert len(cpa_id_list) == len(set(cpa_id_list))

    cpa_dict = {}
    for table_list in nested_list:
        cpa_id = table_list[0]
        cpa_dict[cpa_id] = dict()
        assert len(keys) == len(table_list)
        for key, ele in zip(keys, table_list):
            cpa_dict[cpa_id][key] = ele

    return cpa_dict

def cell_tokenization(task_table_fn, save_fn):

    task_table = load(task_table_fn)
    task_tokenized_table= dict()
    for table_id, table_list in task_table.items():
        task_tokenized_table[table_id] = []
        for row in table_list:
            cell_values = [str(value) if value else "null" for key, value in row.items()]
            cell_lm_indices = tokenizer(
                cell_values, padding=False, truncation=False, max_length=None,
                is_split_into_words=False, return_tensors='np', return_length=True
            )['input_ids'].tolist()
            assert len(cell_lm_indices) == len(row.keys())
            task_tokenized_table[table_id].append(copy.deepcopy(cell_lm_indices))
    save(save_fn, task_tokenized_table)

def table_info():

    dataset_path = "/apollo/users/dya/dataset/wikitables"

    type_vocab = get_path(dataset_path, 'CTA_TURL_LABEL_TXT', 'turl')
    rel_vocab = get_path(dataset_path, 'CPA_TURL_LABEL_TXT', 'turl')

    type_id2label, type_label2id = read_vocab(type_vocab)
    rel_id2label, rel_label2id = read_vocab(rel_vocab)

    cta_pkl_annotation_fn = os.path.join(dataset_path, "pkl", "CTA-TURL.pkl")
    cpa_pkl_annotation_fn = os.path.join(dataset_path, "pkl", "CPA-TURL.pkl")
    cta_pkl_table_fn = os.path.join(dataset_path, "pkl", "cta_table.pkl")
    cpa_pkl_table_fn = os.path.join(dataset_path, "pkl", "cpa_table.pkl")
    cta_pkl_table_lm_tokenized_fn = os.path.join(dataset_path, "pkl", "cta.pkl")
    cpa_pkl_table_lm_tokenized_fn = os.path.join(dataset_path, "pkl", "cpa.pkl")

    cta_dict = load(cta_pkl_annotation_fn)
    cpa_dict = load(cpa_pkl_annotation_fn)

    cta_dict['label2idx'] = type_label2id
    cta_dict['idx2label'] = type_id2label
    cpa_dict['label2idx'] = rel_label2id
    cpa_dict['idx2label'] = rel_id2label

    save(cta_pkl_annotation_fn, cta_dict)
    save(cpa_pkl_annotation_fn, cpa_dict)

    cta_table = {}
    cpa_table = {}
    cta_dict = {'train':None, 'dev':None, 'test':None, 'label2idx':type_label2id, 'idx2label':type_id2label }
    cpa_dict = {'train':None, 'dev':None, 'test':None, 'label2idx':rel_label2id, 'idx2label':rel_id2label }

    for split in ['train', 'dev', 'test']:
        # TODO
        fn_table = os.path.join(dataset_path, "{}_tables.jsonl".format(split))
        fn_cta = os.path.join(dataset_path, "{}.table_col_type.json".format(split))
        fn_cpa = os.path.join(dataset_path, "{}.table_rel_extraction.json".format(split))

        cta_dict[split], cta_table_n = extract_from_raw_table(
            read_table(fn_table), read_cta(fn_cta), type_label2id, task='cta'
        )

        assert len(set(cta_table.keys()) & set(cta_table_n.keys())) == 0
        cta_table.update(cta_table_n)

        cpa_dict[split], cpa_table_n = extract_from_raw_table(
            read_table(fn_table), read_cpa(fn_cpa), rel_label2id, task='cpa'
        )
        assert len(set(cpa_table.keys()) & set(cpa_table_n.keys())) == 0
        cpa_table.update(cpa_table_n)

    save(cta_pkl_annotation_fn, cta_dict)
    save(cta_pkl_table_fn, cta_table)
    save(cpa_pkl_annotation_fn, cpa_dict)
    save(cpa_pkl_table_fn, cpa_table)

    cell_tokenization(cta_pkl_table_fn, cta_pkl_table_lm_tokenized_fn)
    cell_tokenization(cpa_pkl_table_fn, cpa_pkl_table_lm_tokenized_fn)

def main():
    table_info()

if __name__ == "__main__":
    main()