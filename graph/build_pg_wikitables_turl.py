import os
import ast
import csv
import json
import pickle
import pprint
import copy
import neo4j
from neo4j import GraphDatabase
import numpy as np
import pprint
import random
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_seqs

from utils.file_registry import get_path

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def read_vocab(fn):
    topic_dict = dict()
    with open(fn, 'r') as fp:
        for line in fp.readlines():
            if len(line.strip()) == 0:
                continue
            label = line.split('\t')[1].strip()
            topic = label.split('.')[0]
            if topic not in topic_dict:
                topic_dict[topic] = [label]
            else:
                topic_dict[topic].append(label)
    return topic_dict

keys = ['_id', 'pgTitle', 'pgId', 'sectionTitle', 'tableCaption', 'processed_tableHeaders', 'tableData', 'label']

def read_cta(fn_list):

    cta_dict_total = dict()
    for fn in fn_list:

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

        cta_dict_total.update(cta_dict)

    return cta_dict_total

def read_cpa(fn_list):

    cpa_dict_total = dict()
    for fn in fn_list:

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

        cpa_dict_total.update(cpa_dict)

    return cpa_dict_total

def extract_table_data(table_dict):
    textData = []
    for row in table_dict['tableData']:
        textData.append({ str(col_id) : cell['text'] for col_id, cell in enumerate(row)})
    return textData

def read_table(fn_list):

    table_dicts_total = dict()

    for fn in fn_list:
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
        table_dicts_total.update(table_dicts)

    return table_dicts_total

def extract_idx_from_raw_table(table_dicts, task_dict, target_table_ids, save_fn, task):

    output_dict = dict()

    for id, task_data in task_dict.items():

        print('table:', id)
        output_dict[id] = dict()
        output_dict[id].update(task_dict[id])

        if task == 'cpa':

            raw_data = table_dicts[id]
            subj_col_idx = raw_data['subject_column']

            # modify original tables
            tmp_list_2 = []
            for col_idx in range(len(raw_data['table'][0])):
                col_cell_values = [row[str(col_idx)] for row in raw_data['table']]
                col_text = ' '.join(col_cell_values)
                tmp_list_2.append(col_text)

            # find the indexes of task columns from modified tables
            target_col_idx = []
            for col in task_data['tableData']:
                col_text = ' '.join([cell[1][1] for cell in col])
                scores = normalized_damerau_levenshtein_distance_seqs(col_text, tmp_list_2)
                for tmp in np.argsort(scores).tolist():
                    if col[0][1][1] != raw_data['table'][col[0][0][0]][str(tmp)]:
                        continue
                    else:
                        target_col_idx.append(tmp)
                        break

            target_col_idx_pair = []
            for obj_idx in target_col_idx:
                if subj_col_idx == obj_idx:
                    continue
                target_col_idx_pair.append( [subj_col_idx, obj_idx] )

            output_dict[id].update({'col_idx_pair': target_col_idx_pair})

        elif task == 'cta':
            # original tables
            tmp_list = []
            raw_data = table_dicts[id]
            for col_idx in range(len(raw_data['table'][0])):
                col_cell_values = [row[str(col_idx)] for row in raw_data['table']]
                col_text = ' '.join(col_cell_values)
                tmp_list.append(col_text)

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

            assert len(target_col_idx) == len(task_data['tableData'])
            output_dict[id].update({'col_idx': target_col_idx})

        else:
            raise(" Task name must be wrong ! ")

    save(save_fn, output_dict)
    return output_dict

def list_subj_candidate_per_topic(cta_dict, cpa_dict):

    result = dict()

    for table_id, cpa_table_dict in cpa_dict.items():

        subj_idx = cpa_table_dict['col_idx_pair'][0][0]

        if subj_idx in cta_dict[table_id]['col_idx']:

            subj_idx_in_cta_dict = cta_dict[table_id]['col_idx'].index(subj_idx)
            subj_types = cta_dict[table_id]['label'][subj_idx_in_cta_dict]

            for subj_type in subj_types:
                topic = subj_type.split('.')[0]
                if topic not in result:
                    result[topic] = dict()
                if subj_type not in result[topic]:
                    result[topic][subj_type] = 1
                else:
                    result[topic][subj_type] += 1
        else:
            raise(" Could not find subject column's type in this table. ")

    pprint.pprint(result)
    return result

def construct_SPO(cta_dict, cpa_dict):

    SPO = []

    for table_id, cpa_table_dict in cpa_dict.items():

        if len(cpa_table_dict['label']) != len(cpa_table_dict['col_idx_pair']):
            continue

        for i, (subj_idx, obj_idx) in enumerate(cpa_table_dict['col_idx_pair']):

            for predicate in cpa_table_dict['label'][i]:

                if subj_idx in cta_dict[table_id]['col_idx'] and obj_idx in cta_dict[table_id]['col_idx']:

                    p_topic = predicate.split('.')[0]

                    subj_idx_in_cta_dict = cta_dict[table_id]['col_idx'].index(subj_idx)
                    subj_types = cta_dict[table_id]['label'][subj_idx_in_cta_dict]

                    obj_idx_in_cta_dict = cta_dict[table_id]['col_idx'].index(obj_idx)
                    obj_types = cta_dict[table_id]['label'][obj_idx_in_cta_dict]

                    if len(obj_types) == 1:
                        if len(subj_types) == 1:
                            SPO.append([ subj_types[0] , predicate, obj_types[0] ])
                        else:
                            obj_type_topic = obj_types[0].split('.')[0]
                            for subj_type in subj_types:
                                subj_type_topic = subj_type.split('.')[0]
                                if p_topic == subj_type_topic:
                                    SPO.append([subj_type, predicate, obj_types[0]])
                                if obj_type_topic == subj_type_topic:
                                    SPO.append([subj_type, predicate, obj_types[0]])
                    else:
                        if len(subj_types) == 1:
                            subj_type_topic = subj_types[0].split('.')[0]
                            for obj_type in obj_types:
                                obj_type_topic = obj_type.split('.')[0]
                                if p_topic == obj_type_topic:
                                    SPO.append([ subj_types[0], predicate, obj_type ])
                                if subj_type_topic == obj_type_topic:
                                    SPO.append([ subj_types[0], predicate, obj_type ])

                        else:
                            for subj_type in subj_types:
                                for obj_type in obj_types:
                                    subj_type_topic = subj_type.split('.')[0]
                                    obj_type_topic = obj_type.split('.')[0]

                                    if p_topic == subj_type_topic and p_topic == obj_type_topic:
                                        SPO.append([subj_type, predicate, obj_type])

                                    if subj_type_topic == obj_type_topic:
                                        SPO.append([subj_type, predicate, obj_type])
                else:
                    continue

    spo_text = list(set([ s + '!'+ p + '!' + o for s,p,o in SPO]))
    SPO_final = [ text.replace('.', '|').split('!') for text in spo_text ]

    return SPO_final

def construct_SPO_within_one_single_column(cta_dict):

    SPO = []
    unique_topic_group = list()
    example_dict = dict()
    for table_id, info in cta_dict.items():
        for labels in info['label']:
            topics = set([ label.split('.')[0] for label in labels])
            if len(topics) >= 2:
                unique_topic_group_text = ' - '.join(topics)
                unique_topic_group.append(unique_topic_group_text)
                if unique_topic_group_text not in example_dict:
                    example_dict[unique_topic_group_text] = [ labels ]
                else:
                    example_dict[unique_topic_group_text].append(labels)

    duo = dict()
    for table_id, info in cta_dict.items():
        for labels in info['label']:
            if len(labels) == 2:
                unique_duo = ' - '.join(sorted(labels))
                if unique_duo not in duo:
                    duo[unique_duo] = 1
                else:
                    duo[unique_duo] += 1
                SPO.append( [sorted(labels)[0], 'apposition', sorted(labels)[1]] )

    spo_text = list(set([ s + '!'+ p + '!' + o for s,p,o in SPO]))
    SPO_final = [ text.replace('.', '|').split('!') for text in spo_text ]

    return SPO_final

def create_pg_neo4j(SPO):

    # step 1: connect to DB
    URI = "neo4j+s://"
    AUTH = ("neo4j", "")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()

    ndata = set( [ spo[0] for spo in SPO ] + [ spo[2] for spo in SPO ] )

    # # step 2: create nodes
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        nmap = {}
        for i, node in enumerate(ndata):

            for symbol in ['?', '+']:
                if symbol in node:
                    nname = node.split(symbol)[-1]
                else:
                    nname = node.split('/')[-1]
            nname = ''.join(filter(str.isalnum, nname)) + '_{}'.format(i)
            nmap[node] = nname

            create_node_query = ("CREATE (node:%s { label: $label, source: $source })" % (nname))
            create_node_parameters = {
                "label": nname,
                "source": "'" + node + "'",
            }
            record = driver.execute_query(
                query_=create_node_query,
                parameters_=create_node_parameters,
                routing_=neo4j.RoutingControl.WRITE,
                database_="neo4j",
            )

    # step 3: create edges
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        emap = {}
        for s, p, o in SPO:

            for symbol in ['?', '+']:
                if symbol in node:
                    ename = p.split(symbol)[-1]
                else:
                    ename = p.split('/')[-1]
            ename =''.join(filter(str.isalnum, ename))
            emap[p] = ename

            create_relationship_query = (
                    "MATCH (node1:{})  ".format(nmap[s]) +
                    "MATCH (node2:{})  ".format(nmap[o]) +
                    "CREATE (node1) -[r:%s { label:\"%s\" }]-> (node2)  " % (ename, p) +
                    "RETURN r"
            )
            print(create_relationship_query)
            record = driver.execute_query(
                query_=create_relationship_query,
                routing_=neo4j.RoutingControl.WRITE,
                database_="neo4j",
            )

def table_gt_info(dataset_path=None):

    table_train_path = get_path(dataset_path, 'TURL_TRAIN_TABLE_JSON', 'turl')
    table_val_path = get_path(dataset_path, 'TURL_DEV_TABLE_JSON', 'turl')

    node_train_gt_path = get_path(dataset_path, 'CTA_TURL_TRAIN_GT_JSON', 'turl')
    node_val_gt_path = get_path(dataset_path, 'CTA_TURL_DEV_GT_JSON', 'turl')
    edge_train_gt_path = get_path(dataset_path, 'CPA_TURL_TRAIN_GT_JSON', 'turl')
    edge_val_gt_path = get_path(dataset_path, 'CPA_TURL_DEV_GT_JSON', 'turl')

    cta_dict = read_cta([node_train_gt_path, node_val_gt_path])
    cpa_dict = read_cpa([edge_train_gt_path, edge_val_gt_path])
    table_dict = read_table([table_train_path, table_val_path])
    target_table_ids = set(cta_dict.keys()) & set(cpa_dict.keys())

    cta_selected_dict_fn = get_path(dataset_path, 'CTA_PG', 'turl')
    cpa_selected_dict_fn = get_path(dataset_path, 'CPA_PG', 'turl')

    extract_idx_from_raw_table(table_dict, cta_dict, target_table_ids, cta_selected_dict_fn, 'cta')
    extract_idx_from_raw_table(table_dict, cpa_dict, target_table_ids, cpa_selected_dict_fn, 'cpa')

    cta_selected_dict = load(cta_selected_dict_fn)
    cpa_selected_dict = load(cpa_selected_dict_fn)

    SPO = construct_SPO(cta_selected_dict, cpa_selected_dict)
    SPO_duo = construct_SPO_within_one_single_column(cta_selected_dict)
    SPO_final = SPO + SPO_duo
    for i, spo in enumerate(SPO_final):
        s,p,o = spo
        SPO_final[i] = [ s.replace('|', '.'), p.replace('|', '.'), o.replace('|', '.') ]

    return SPO_final

def main():
    dataset_path = "/apollo/users/dya/dataset/wikitables"
    SPO = table_gt_info(dataset_path)
    # create_pg_neo4j(SPO)

if __name__ == "__main__":
    main()