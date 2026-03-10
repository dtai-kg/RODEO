import os
import csv
import json
import pprint
from collections import defaultdict, Counter

from utils.file_registry import get_path

def construct_type_lookup(file_paths):

    type_lookup = dict()
    
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)
            print("Header:", header) # Header: ['table_name', 'column_index', 'label']
            for row in csv_reader:
                raw_table_name, column_index, type_label = \
                    row[0].strip(), row[1].strip(), row[2].strip()
                table_name = raw_table_name.split("_CTA.json.gz")[0]
                key = f"{table_name}-*-{column_index}"
                if key in type_lookup:
                    raise ValueError(f"Duplicate key {key}.")
                type_lookup[key] = type_label

    return type_lookup 

def construct_alternative_subj_type_lookup(file_paths, type_lookup):

    cta_table_names = set([ key.split('-*-')[0] for key in type_lookup])
    topic2S = defaultdict(list)
    
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)
            for row in csv_reader:
                raw_table_name, subj_column_index = row[0].strip(), row[1].strip()
                table_name = raw_table_name.split("_CPA.json.gz")[0]
                topic = raw_table_name.split('_')[0] 
                if table_name not in cta_table_names:
                    continue
                subj_type = fixed_index_type_lookup(f"{table_name}-*-{subj_column_index}", "subj", type_lookup)
                topic2S[topic].append(subj_type)

    topic2S = { k: ([x for x in v if x is not None] or [f"?{k}"]) for k, v in topic2S.items()}
    topic2S = { k: Counter(v).most_common(1)[0][0] if v else None for k, v in topic2S.items()}

    return topic2S

def add_synthetic_links(file_paths, SPO, alternative_subj_type_lookup):

    tmp = set()
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)
            print("Header:", header) # Header: ['table_name', 'column_index', 'label']
            for row in csv_reader:
                raw_table_name, column_index, type_label = \
                    row[0].strip(), row[1].strip(), row[2].strip()
                topic = raw_table_name.split('_')[0] 
                subj_type = alternative_subj_type_lookup[topic]
                if type_label != subj_type:
                    tmp.add((subj_type, '??'+ type_label.split('/')[-1], type_label))

    SO = [ s+o for s,p,o in SPO]
    for s, p, o in tmp:
        if s+o not in SO:
            SPO.add((s,p,o))
            
    return SPO
            
                
def construct_SPO(file_paths, type_lookup, alternative_subj_type_lookup):

    SPO_multiple_set = list()
    cta_table_names = set([ key.split('-*-')[0] for key in type_lookup])
    
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)
            print("Header:", header) # Header: ['table_name', 'main_column_index', 'column_index', 'label']
            for row in csv_reader:
                raw_table_name, subj_column_index, obj_column_index, property_label = \
                    row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                table_name = raw_table_name.split("_CPA.json.gz")[0]

                subj_type = fixed_index_type_lookup( f"{table_name}-*-{subj_column_index}", "subj", type_lookup, alternative_subj_type_lookup )
                obj_type = fixed_index_type_lookup( f"{table_name}-*-{obj_column_index}", "obj", type_lookup )

                if subj_type == obj_type: continue
                if obj_type is None: obj_type = f"?{property_label.split('/')[-1]}"

                SPO_multiple_set.append((subj_type, property_label, obj_type))

    SP = set()
    PO = defaultdict(list)
    for s, p, o in SPO_multiple_set:
        SP.add((s, p))
        PO[p].append(o)

    for p, O in PO.items():
        tmp = [ o for o in O if '?' not in o]
        if len(tmp) != 0:
            PO[p] = Counter(tmp).most_common(1)[0][0]
        else:
            PO[p] = O[0]

    SPO = set()
    for s,p in SP:
        SPO.add((s,p,PO[p]))
    
    return SPO

def fixed_index_type_lookup(key, usage, type_lookup, alternative_subj_type_lookup=None):

    dataset_path = "/apollo/users/dya/dataset/semtab"

    # CTA_Tables and CPA_Tables are placeholders
    CTA_DATASET = os.path.join(dataset_path, "CTA_Tables")
    CPA_DATASET = os.path.join(dataset_path, "CPA_Tables")

    table_name = key.split('-*-')[0]
    column_index = key.split('-*-')[1]

    cta_table = os.path.join(CTA_DATASET, table_name + '_CTA.json')
    cpa_table = os.path.join(CPA_DATASET, table_name + '_CPA.json')

    def read_first_json_line(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(next(f))

    if not os.path.exists(cta_table):
        pass
    else:
        _dict_cpa = read_first_json_line(cpa_table)
        _dict_cta = read_first_json_line(cta_table)
        
        cpa_value = _dict_cpa[column_index]
        cta_value = _dict_cta[column_index]
        
        if cpa_value == cta_value: 
            key = key
        else:
            new_column_index = next((k for k, v in _dict_cta.items() if v == cpa_value), None)
            key = f"{table_name}-*-{new_column_index}"

    if usage == "subj":
        if alternative_subj_type_lookup:
            topic = table_name.split('_')[0] 
            return alternative_subj_type_lookup[topic]
        else:
            if key in type_lookup:
                return type_lookup[key]
            else:
                return None
            
    if usage == "obj":
        if key in type_lookup:
            return type_lookup[key]
        else:
            return None

def table_gt_info(dataset_path=None):

    cta_train_gt_path = get_path(dataset_path, 'CTA_DBP_TRAIN_GT_CSV', 'sotab')
    cta_val_gt_path = get_path(dataset_path, 'CTA_DBP_VAL_GT_CSV', 'sotab')
    cpa_train_gt_path = get_path(dataset_path, 'CPA_DBP_TRAIN_GT_CSV', 'sotab')
    cpa_val_gt_path = get_path(dataset_path, 'CPA_DBP_VAL_GT_CSV', 'sotab')

    type_lookup = construct_type_lookup([cta_train_gt_path, cta_val_gt_path])
    alternative_subj_type_lookup = construct_alternative_subj_type_lookup([cpa_train_gt_path, cpa_val_gt_path], type_lookup) 
    SPO = construct_SPO([cpa_train_gt_path, cpa_val_gt_path], type_lookup, alternative_subj_type_lookup)
    SPO = add_synthetic_links([cta_train_gt_path, cta_val_gt_path], SPO, alternative_subj_type_lookup)

    return SPO, { k: {'subj': v} for k, v in alternative_subj_type_lookup.items()}


def main():
    dataset_path = "/apollo/users/dya/dataset/semtab"
    SPO, topic2S = table_gt_info(dataset_path)

if __name__ == "__main__":
    main()
