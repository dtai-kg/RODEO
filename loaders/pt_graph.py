import os
os.environ["DGLBACKEND"] = "pytorch"
import pprint

import re
import dgl
import numpy as np
import torch
import torch.nn as nn

from utils.data_utils import load
from utils.file_registry import get_path

def read_label(filename, _format='csv', _type='node', _source='dbpedia', _table='sotab'):
    with open(filename, 'r') as file:
        data = {'type': _type, 'source': _source, 'table': _table, 'idx2label': {}, 'label2idx': {} }
        if _table in ['sotab', 'gittab'] :
            for _id, row in enumerate(file.readlines()):
                if row.strip() != 0:
                    data['idx2label'][_id] = row.strip()
                    data['label2idx'][row.strip()] = _id
        elif _table == 'wiki_table':
            for row in file.readlines():
                if len(row.split()) == 2:
                    items = row.split()
                    data['idx2label'][int(items[0])] = items[1]
                    data['label2idx'][items[1].strip()] = int(items[0])
        else:
            raise ValueError('Could not find the dataset label.')

    # print (data)
    return data

# node -> multiple types
# relationship -> multiple types
def load_spo2dgl(SPO, topic2S, cta_fn, cpa_fn, _source, _table):

    assert _table == 'sotab'

    # Step 1: read CTA/CPA vocab files in different datasets
    ndata = read_label(cta_fn, _format='txt', _type='node', _source=_source, _table=_table)
    edata = read_label(cpa_fn, _format='txt', _type='edge', _source=_source, _table=_table)

    # Step 2: indexing new-added node/edge
    added_nodes = set()
    added_edges = set()
    for s, p, o in SPO:
        if s not in ndata['label2idx']:
            added_nodes.add(s)
        if o not in ndata['label2idx']:
            added_nodes.add(o)
        if p not in edata['label2idx']:
            added_edges.add(p)

    next_node_idx = len(ndata['label2idx'])
    added_node_idx_list = list()
    for node in sorted(list(added_nodes)):
        ndata['label2idx'][node] = next_node_idx
        ndata['idx2label'][next_node_idx] = node
        added_node_idx_list.append(next_node_idx)
        next_node_idx += 1

    next_edge_idx = len(edata['label2idx'])
    added_edge_idx_list = list()
    for edge in sorted(list(added_edges)):
        edata['label2idx'][edge] = next_edge_idx
        edata['idx2label'][next_edge_idx] = edge
        added_edge_idx_list.append(next_edge_idx)
        next_edge_idx += 1

    print("CTA vocabulary:")
    print(ndata)
    print("CPA vocabulary:")
    print(edata)

    # Step 3: Create DGL graph
    g = dgl.DGLGraph()

    # Step 4: Add nodes into DGL graph
    g.add_nodes(len(ndata['label2idx']))

    # keep the new added label with '??' in the bottom and filter out them easily
    SPO = sorted(SPO, key=lambda spo: spo[1], reverse=True)

    added_edge_idx_list2 = list()
    for i, (s, p, o) in enumerate(SPO):
        if p[:2] == '??':
            added_edge_idx_list2.append(i)

    # Step 5: Add edges
    src_ids = [ndata['label2idx'][s] for s, p, o in SPO]
    dst_ids = [ndata['label2idx'][o] for s, p, o in SPO]
    rel_ids = [edata['label2idx'][p] for s, p, o in SPO]

    g.add_edges(src_ids, dst_ids)
    g.edata['rel_type'] = torch.LongTensor(rel_ids)
    assert g.edata['rel_type'].shape[0] == g.number_of_edges()

    # Step 6: Save the dict to use the right edge
    # one property may have multiple edges (they may link different subjects and objects)
    # during the training, each table has one topic -> each topic has only one subject
    # -> based on the topic/subject, we determine the edge (given the same property)
    p_dict_by_topic = {}
    S2topic = {value['subj']: key for key, value in topic2S.items()}

    for p1 in set(rel_ids):
        p_dict_by_topic[p1] = dict()
        for X, (s, p2, o) in enumerate(SPO):
            if p1 == edata['label2idx'][p2]:
                topic = S2topic[s]
                if topic in p_dict_by_topic[p1]:
                    print(p_dict_by_topic[p1])
                    print(s, p2, o)
                    assert p_dict_by_topic[p1][topic] == X
                if s == o:
                    print(s, p2, o)
                p_dict_by_topic[p1][topic] = X

    added_label_ids = {'node': added_node_idx_list, 'edge': added_edge_idx_list, 'edge_gnn': added_edge_idx_list2}

    return g, ndata, edata, added_label_ids, p_dict_by_topic


def load_spo2dgl_turl(SPO, p_spo_dict, cta_fn, cpa_fn, _source, _table):

    # Step 1: read CTA/CPA vocab files in different datasets
    ndata = read_label(cta_fn, _format='txt', _type='node', _source=_source, _table=_table)
    edata = read_label(cpa_fn, _format='txt', _type='edge', _source=_source, _table=_table)

    # Step 2: indexing new-added node/edge
    added_nodes = set()
    added_edges = set()

    # remove subproperty triples
    # add ?? to type
    SPO = [[s, '??rdf.type', o] if 'rdf.type' in p else [s, p, o] for s, p, o in SPO]

    for s, p, o in SPO:
        if s not in ndata['label2idx']:
            added_nodes.add(s)
        if o not in ndata['label2idx']:
            added_nodes.add(o)
        if p not in edata['label2idx']:
            added_edges.add(p)

    next_node_idx = len(ndata['label2idx'])
    added_node_idx_list = list()
    for node in sorted(list(added_nodes)):
        ndata['label2idx'][node] = next_node_idx
        ndata['idx2label'][next_node_idx] = node
        added_node_idx_list.append(next_node_idx)
        next_node_idx += 1

    next_edge_idx = len(edata['label2idx'])
    added_edge_idx_list = list()
    for edge in sorted(list(added_edges)):
        edata['label2idx'][edge] = next_edge_idx
        edata['idx2label'][next_edge_idx] = edge
        added_edge_idx_list.append(next_edge_idx)
        next_edge_idx += 1

    # keep the new added label with '??' at the bottom and filter out them easily
    SPO = sorted(SPO, key=lambda spo: spo[1], reverse=True)
    # Create the dictionary
    SPO_dict = {}
    for s, p, o in SPO:
        if p not in SPO_dict:
            SPO_dict[p] = []
        SPO_dict[p].append([s, p, o])

    added_edge_idx_list2 = list()
    for i, (s, p, o) in enumerate(SPO):
        if p[:2] == '??':
            added_edge_idx_list2.append(i)

    print("CTA vocabulary:")
    print(ndata)
    print("CPA vocabulary:")
    print(edata)

    # Step 3: Create DGL graph
    g = dgl.DGLGraph()

    # Step 4: Add nodes into DGL graph
    g.add_nodes(len(ndata['label2idx']))

    # Step 5: Add edges
    src_ids = [ndata['label2idx'][s] for s, p, o in SPO]
    dst_ids = [ndata['label2idx'][o] for s, p, o in SPO]
    rel_ids = [edata['label2idx'][p] for s, p, o in SPO]
    g.add_edges(src_ids, dst_ids)
    g.edata['rel_type'] = torch.LongTensor(rel_ids)
    assert g.edata['rel_type'].shape[0] == g.number_of_edges()

    # not in use
    p_positive_dict = dict()
    for pi, pp in enumerate(rel_ids):
        if pp not in p_positive_dict:
            p_positive_dict[pp] = list()
        p_positive_dict[pp].append(pi)

    dataset_path = "/apollo/users/dya/dataset/wikitables"

    cta_dict_path = get_path(dataset_path, 'CTA_PG', 'turl')
    cpa_dict_path = get_path(dataset_path, 'CPA_PG', 'turl')

    cta_dict = load(cta_dict_path)
    cpa_dict = load(cpa_dict_path)

    subj_dict_path = get_path(dataset_path, 'SUBJ_DICT', 'turl')
    obj_dict_path = get_path(dataset_path, 'OBJ_DICT', 'turl')
    subj_dict = load(subj_dict_path)
    obj_dict = load(obj_dict_path)

    p_dict_by_table = dict()
    for table_id, cpa_info in cpa_dict.items():
        if table_id in cta_dict:
            cta_info = cta_dict[table_id]
            cta_labels = cta_info['label']
            cta_col_idx = cta_info['col_idx']
            for cpa_labels, cpa_col_idx_pair in zip(cpa_info['label'], cpa_info['col_idx_pair']):
                try:
                    col1_types = cta_labels[cta_col_idx.index(cpa_col_idx_pair[0])]
                except:
                    col1_types = list(set([subj_dict[tmp] for tmp in cpa_labels]))

                try:
                    col2_types = cta_labels[cta_col_idx.index(cpa_col_idx_pair[1])]
                except:
                    col2_types = list(set([obj_dict[tmp] for tmp in cpa_labels]))

                for p in cpa_labels:

                    if p not in p_dict_by_table:
                        p_dict_by_table[p] = dict()
                    p_dict_by_table[p][table_id] = list()

                    for s, _, o in p_spo_dict[p]:
                        if [s, p, o] not in SPO_dict[p]:
                            if [o, p, s] in SPO_dict[p]:
                                spo = [o, p, s]
                            else:
                                continue
                        else:
                            spo = [s, p, o]

                        if s in col1_types and o in col2_types:
                            p_dict_by_table[p][table_id].append(spo)
                        elif s in col2_types and o in col1_types:
                            p_dict_by_table[p][table_id].append(spo)
                        else:
                            continue

                    if len(p_dict_by_table[p][table_id]) == 0:
                        p_dict_by_table[p][table_id].append([subj_dict[p], p, obj_dict[p]])

        else:
            for cpa_labels, cpa_col_idx_pair in zip(cpa_info['label'], cpa_info['col_idx_pair']):

                col1_types = [subj_dict[tmp3] for tmp3 in cpa_labels]
                col2_types = [obj_dict[tmp3] for tmp3 in cpa_labels]

                for p in cpa_labels:
                    if p not in p_dict_by_table:
                        p_dict_by_table[p] = dict()
                    p_dict_by_table[p][table_id] = list()

                    for s, _, o in p_spo_dict[p]:
                        if [s, p, o] not in SPO_dict[p]:
                            if [o, p, s] in SPO_dict[p]:
                                spo = [o, p, s]
                            else:
                                continue
                        else:
                            spo = [s, p, o]
                        if s in col1_types and o in col2_types:
                            p_dict_by_table[p][table_id].append(spo)
                        elif s in col2_types and o in col1_types:
                            p_dict_by_table[p][table_id].append(spo)
                        else:
                            continue
                    if len(p_dict_by_table[p][table_id]) == 0:
                        p_dict_by_table[p][table_id].append([subj_dict[p], p, obj_dict[p]])

    # find edge index
    p_dict_by_table2 = dict()
    for p in p_dict_by_table:
        p_dict_by_table2[edata['label2idx'][p]] = dict()
        for table_id, spo_list in p_dict_by_table[p].items():
            p_dict_by_table2[edata['label2idx'][p]][table_id] = [SPO.index(spo) for spo in spo_list]

    added_label_ids = {'node': added_node_idx_list, 'edge': added_edge_idx_list, 'edge_gnn': added_edge_idx_list2}

    return g, ndata, edata, added_label_ids, p_dict_by_table2, p_positive_dict

# node -> multiple types
# relationship -> multiple types
def load_gittab_spo2dgl(SPO, cta_fn, cpa_fn, _source, _table):

    assert _table == 'gittab'

    # Step 1: read CTA/CPA vocab files in different datasets
    ndata = read_label(cta_fn, _format='txt', _type='node', _source=_source, _table=_table)
    edata = read_label(cpa_fn, _format='txt', _type='edge', _source=_source, _table=_table)

    # Step 2: indexing new-added node/edge
    added_nodes = set()
    added_edges = set()
    for s, p, o in SPO:
        if s not in ndata['label2idx']:
            added_nodes.add(s)
        if o not in ndata['label2idx']:
            added_nodes.add(o)
        if p not in edata['label2idx']:
            added_edges.add(p)

    # print (added_nodes)
    # print (added_edges)

    def natural_key(s):
        """Key for natural sorting — handles numeric parts."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    next_node_idx = len(ndata['label2idx'])
    added_node_idx_list = list()
    for node in sorted(list(added_nodes), key=natural_key):
        ndata['label2idx'][node] = next_node_idx
        ndata['idx2label'][next_node_idx] = node
        added_node_idx_list.append(next_node_idx)
        next_node_idx += 1

    next_edge_idx = len(edata['label2idx'])
    added_edge_idx_list = list()
    for edge in sorted(list(added_edges), key=natural_key):
        edata['label2idx'][edge] = next_edge_idx
        edata['idx2label'][next_edge_idx] = edge
        added_edge_idx_list.append(next_edge_idx)
        next_edge_idx += 1

    # print(added_node_idx_list)
    # print(added_edge_idx_list)

    print("CTA vocabulary:")
    print(ndata)
    print("CPA vocabulary:")
    print(edata)

    # Step 3: Create DGL graph
    g = dgl.DGLGraph()

    # Step 4: Add nodes into DGL graph
    g.add_nodes(len(ndata['label2idx']))

    SPO = sorted(SPO, key=lambda spo: natural_key(spo[1]), reverse=False)

    added_edge_idx_list2 = list()

    # Step 5: Add edges
    src_ids = [ndata['label2idx'][s] for s, p, o in SPO]
    dst_ids = [ndata['label2idx'][o] for s, p, o in SPO]
    rel_ids = [edata['label2idx'][p] for s, p, o in SPO]

    g.add_edges(src_ids, dst_ids)
    g.edata['rel_type'] = torch.LongTensor(rel_ids)
    assert g.edata['rel_type'].shape[0] == g.number_of_edges()

    # Step 6: Save the dict to use the right edge
    # one property may have multiple edges (they may link different subjects and objects)
    p_dict_by_topic = {}

    for p1 in set(rel_ids):
        p_dict_by_topic[p1] = dict()
        for X, (s, p2, o) in enumerate(SPO):
            if p1 == edata['label2idx'][p2]:
                topic = s
                if topic in p_dict_by_topic[p1]:
                    print(p_dict_by_topic[p1])
                    print(s, p2, o)
                    assert p_dict_by_topic[p1][topic] == X
                p_dict_by_topic[p1][topic] = X

    print (p_dict_by_topic)

    added_label_ids = {'node': added_node_idx_list, 'edge': added_edge_idx_list, 'edge_gnn': added_edge_idx_list2}

    return g, ndata, edata, added_label_ids, p_dict_by_topic

def create_graph(args):

    assert len(set([task[4:] for task in args.tasks])) == 1, "Found: Configuration with more than one dataset"

    source = [ task[4:] for task in args.tasks ][0]

    if 'GIT' in source:

        dataset_path = args.gittab_dataset_path
        cta_fn = get_path(dataset_path, 'CTA_GIT_LABEL_TXT', dataset='gittab')
        cpa_fn = get_path(dataset_path, 'SYNTHETIC_REL_LABEL_TXT', dataset='gittab')

        from graph.build_pg_gittab_dbpedia import table_gt_info
        SPO, _ = table_gt_info(dataset_path)

        g, nlabel, elabel, added_label_ids, p_dict = load_gittab_spo2dgl(SPO, cta_fn, cpa_fn, _source='dbpedia', _table='gittab')
        return g, nlabel, elabel, added_label_ids, p_dict

    elif 'TURL' in source:

        dataset_path = args.turl_dataset_path
        cta_fn = get_path(dataset_path, 'CTA_TURL_LABEL_TXT', 'turl')
        cpa_fn = get_path(dataset_path, 'CPA_TURL_LABEL_TXT', 'turl')

        spo_fn = get_path(dataset_path, 'TRIPLE_PKL', 'turl')
        # remove duplicated triple
        SPO = list(map(list, set(map(tuple, load(spo_fn)))))

        p_fn = get_path(dataset_path, 'P_SPO_DICT_PKL', 'turl')
        p_spo_dict = load(p_fn)

        g, nlabel, elabel, added_label_ids, p_dict, p_positive_dict = load_spo2dgl_turl(SPO, p_spo_dict, cta_fn, cpa_fn, _source='freebase', _table='wiki_table')
        return g, nlabel, elabel, added_label_ids, p_dict, p_positive_dict

    elif 'DBP' in source:

        dataset_path = args.sotab_dataset_path
        cta_fn = get_path(dataset_path, 'CTA_DBP_LABEL_TXT', 'sotab')
        cpa_fn = get_path(dataset_path, 'CPA_DBP_LABEL_TXT', 'sotab')

        from graph.build_pg_sotab_dbpedia import table_gt_info
        SPO, topic2S= table_gt_info(dataset_path)

        g, nlabel, elabel, added_label_ids, p_dict = load_spo2dgl(SPO, topic2S, cta_fn, cpa_fn, _source='dbpedia', _table='sotab')
        return g, nlabel, elabel, added_label_ids, p_dict

    elif 'SCH' in source:

        dataset_path = args.sotab_dataset_path
        cta_fn = get_path(dataset_path, 'CTA_SCH_LABEL_TXT', 'sotab')
        cpa_fn = get_path(dataset_path, 'CPA_SCH_LABEL_TXT', 'sotab')

        from graph.build_pg_sotab_schema_org import table_gt_info
        SPO, topic2S = table_gt_info(dataset_path)

        g, nlabel, elabel, added_label_ids, p_dict = load_spo2dgl(SPO, topic2S, cta_fn, cpa_fn, _source='schema', _table='sotab')
        return g, nlabel, elabel, added_label_ids, p_dict

    else:
        raise ValueError("task name must be wrong.")

def test():

    from utils.data_utils import load_args_from_parser
    args = load_args_from_parser()

    TASK = ['DBP', 'SCH', 'TURL', 'GIT']

    if 'DBP' in TASK:

        dataset_path = args.sotab_dataset_path
        cta_fn = get_path(dataset_path, 'CTA_DBP_LABEL_TXT', 'sotab')
        cpa_fn = get_path(dataset_path, 'CPA_DBP_LABEL_TXT', 'sotab')

        from graph.build_pg_sotab_dbpedia import table_gt_info
        SPO, topic2S = table_gt_info(dataset_path)

        load_spo2dgl(SPO, topic2S, cta_fn, cpa_fn, _source='dbpedia', _table='sotab')

    if 'SCH' in TASK:

        dataset_path = args.sotab_dataset_path
        cta_fn = get_path(dataset_path, 'CTA_SCH_LABEL_TXT', 'sotab')
        cpa_fn = get_path(dataset_path, 'CPA_SCH_LABEL_TXT', 'sotab')

        from graph.build_pg_sotab_schema_org import table_gt_info
        SPO, topic2S = table_gt_info(dataset_path)

        load_spo2dgl(SPO, topic2S, cta_fn, cpa_fn, _source='schema', _table='sotab')

    if 'GIT' in TASK:

        dataset_path = args.gittab_dataset_path
        cta_fn = get_path(dataset_path, 'CTA_GIT_LABEL_TXT', dataset='gittab')
        cpa_fn = get_path(dataset_path, 'SYNTHETIC_REL_LABEL_TXT', dataset='gittab')

        from graph.build_pg_gittab_dbpedia import table_gt_info
        SPO, _ = table_gt_info(dataset_path)

        load_gittab_spo2dgl(SPO, cta_fn, cpa_fn, _source='dbpedia', _table='gittab')

    if 'TURL' in  TASK:

        dataset_path = args.turl_dataset_path
        cta_fn = get_path(dataset_path, 'CTA_TURL_LABEL_TXT', 'turl')
        cpa_fn = get_path(dataset_path, 'CPA_TURL_LABEL_TXT', 'turl')

        spo_fn = get_path(dataset_path, 'TRIPLE_PKL', 'turl')
        # remove duplicated triple
        SPO = list(map(list, set(map(tuple, load(spo_fn)))))

        p_fn = get_path(dataset_path, 'P_SPO_DICT_PKL', 'turl')
        p_spo_dict = load(p_fn)

        load_spo2dgl_turl(SPO, p_spo_dict, cta_fn, cpa_fn, _source='freebase', _table='wiki_table')

if __name__ == '__main__':
    test()