import os
import copy
import json
import random
import pickle
import argparse
import pprint
from time import time
from collections import Counter

import wandb

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
from torch.nn import PairwiseDistance
import torch.nn.functional as F

from layers.graph_nn import SemanticLabelEncoder
from train.loss import TripletSoftMarginWithDistanceLoss
from utils.data_utils import load_args_from_parser
from loaders.load_dataset import create_loader, create_test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = load_args_from_parser()

def inference():

    # hard coded -> line 79,80 ->  TODO -> update model names

    # write the results to csv files
    write = False

    # iterations: large tables with massive rows and columns => random splitting may lead to different results.
    iteration = 3

    valid_datasets, valid_dataloaders, test_datasets, test_dataloaders, data_dict, task_num_class_dict = create_test_loader(args)
    dist_func = PairwiseDistance(p=2)

    note = args.note
    shortcut_name = args.shortcut_name
    batch_size = args.batch_size
    max_length = args.max_length
    emb_dim = args.emb_dim

    tag_name = "model/{}-bs{}-ml-{}-{}".format(shortcut_name, batch_size, max_length, note)

    loss_fns = [
        TripletSoftMarginWithDistanceLoss(
            distance_function=dist_func,
            margin=1,
            swap=False,
            reduction='mean'
        ) for _ in args.tasks
    ]

    for k, (task, valid_dataset, test_dataset, valid_dataloader,
            test_dataloader, loss_fn) in enumerate(
                zip(args.tasks, valid_datasets, test_datasets,
                    valid_dataloaders, test_dataloaders, loss_fns)):

        if args.shortcut_name in [ "bert-base-multilingual-cased", "bert-base-uncased"]:

            from transformers import BertConfig
            from layers.bert_task_layer import BertForMatching

            model_config = BertConfig.from_pretrained(shortcut_name)
            model = BertForMatching(model_config)
            model.init_by_task(task_num_class_dict, emb_dim)
            model.to(device)

            model.bert.pooler = model.poolers[task].to(device)

            # TODO
            model_filename = "{}={}_marco_cta1_1cpa1_2_200.pt".format(tag_name, task)
            model_l_filename = "{}={}_L_marco_cta1_1cpa1_2_200.pt".format(tag_name, task)
            print (model_filename)
            print (model_l_filename)

            load_path = os.path.join(args.model_save_path, model_filename)
            sd = torch.load(load_path, map_location=device)
            model.load_state_dict(sd)

        else:
            raise ("Could not find / load the language model: {}".format(args.shortcut_name))

        num_unique_nodes = next(task_num_class_dict[task] for task in args.tasks if 'CTA' in task)
        num_unique_edges = next(task_num_class_dict[task] for task in args.tasks if 'CPA' in task)

        model_l = SemanticLabelEncoder({
            'emb_dim': emb_dim,
            'num_unique_nodes': num_unique_nodes,
            'num_unique_edges': num_unique_edges
        })
        model_l.to(device)

        load_path = os.path.join(args.model_save_path, model_l_filename)
        sd_g = torch.load(load_path, map_location=device)
        model_l.load_state_dict(sd_g)

        model.eval()
        model_l.eval()

        for _iter in range(iteration):

            ts_pred_list = []
            ts_tbnames = []
            ts_col_nos = []
            ts_cls_idx = []
            ts_cuts = []

            test_dataset.generate_epoch()
            print("Test data Length: ", len(test_dataset))
            for batch_idx, batch in enumerate(test_dataloader):

                logits, = model(
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                    task = task,
                )

                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(0)

                cls_indexes = batch['cls_idx']
                filtered_logits = torch.zeros(cls_indexes.shape[0], logits.shape[2]).to(device)
                for n in range(cls_indexes.shape[0]):
                    i, j = cls_indexes[n]
                    logit_n = logits[i, j, :]
                    filtered_logits[n] = logit_n

                anchor = F.normalize(filtered_logits, p=2, dim=1)

                vocab_size = task_num_class_dict[task]
                vocab_inputs = torch.tensor([idx for idx in range(vocab_size)], dtype=torch.int).to(device)
                gnn_embeddings = model_l._get_outputs(vocab_inputs, task)
                gnn_embeddings = F.normalize(gnn_embeddings, p=2, dim=1)

                distances = torch.square(torch.cdist(anchor.unsqueeze(1), gnn_embeddings.unsqueeze(0), p=2).squeeze(1))

                ts_cls_idx.append(copy.deepcopy(cls_indexes.cpu().detach().numpy().tolist()))
                ts_pred_list.append(distances.argmin(-1).cpu().detach().numpy().flatten().tolist())

                ts_tbnames.append(batch["tbname"])
                ts_col_nos.append(batch["col_nos"])
                ts_cuts.append(batch["cut"])

            if 'CTA' in task:

                max_col = 10

                cta_ts_true = []
                cta_ts_pred = []
                cta_ts_true_col = []
                cta_ts_pred_col = []
                cta_result_dict = {}

                for batch_i, (pred_list, tbnames, cls_idxs, cuts) in enumerate(zip(ts_pred_list, ts_tbnames, ts_cls_idx, ts_cuts)):

                    assert len(pred_list) == len(cls_idxs)
                    assert len(tbnames) == len(cuts)
                    combos1 = []
                    combos2 = []

                    for tbname, cut in zip(tbnames, cuts):
                        tdata = data_dict[task]['test'][tbname]
                        # tdata = data_dict[task]['validation'][tbname]

                        if cut == 0:
                            cut0 = sorted([(tbname, int(col), l) for cnt, (col, l) in enumerate(zip(tdata['col_idx'], tdata['label'])) if cnt < max_col], key=lambda x: x[1])
                            combos1 += cut0
                        elif cut == 1:
                            cut1 = sorted([(tbname, int(col), l) for cnt, (col, l) in enumerate(zip(tdata['col_idx'], tdata['label'])) if cnt >= max_col], key=lambda x: x[1])
                            combos1 += cut1
                        else:
                            raise ValueError("tables were split twice at most based on columns")

                    combos2 += [(cls_idx[0], p) for cls_idx, p in zip(cls_idxs, pred_list)]

                    assert len(combos1) == len(combos2)

                    for c1,c2 in zip(combos1, combos2):

                        cta_ts_true.append(c1[2])
                        cta_ts_pred.append(c2[1])

                        tbname = c1[0]
                        col_idx = c1[1]
                        predicted_label = c2[1]
                        true_label = c1[2]

                        if tbname not in cta_result_dict:
                            cta_result_dict[tbname] = {}

                        if col_idx in cta_result_dict[tbname]:
                            cta_result_dict[tbname][col_idx]['prediction'].append(predicted_label)
                        else:
                            cta_result_dict[tbname][col_idx] = {}
                            cta_result_dict[tbname][col_idx]['prediction'] = [predicted_label]
                            cta_result_dict[tbname][col_idx]['label'] = true_label

                for tbname, tdata in cta_result_dict.items():
                    for col_idx, item in tdata.items():
                        counter = Counter(cta_result_dict[tbname][col_idx]['prediction'])
                        most_common_value = counter.most_common(1)[0][0]
                        cta_result_dict[tbname][col_idx]['prediction'] = most_common_value

                        cta_ts_true_col.append(cta_result_dict[tbname][col_idx]['label'])
                        cta_ts_pred_col.append(cta_result_dict[tbname][col_idx]['prediction'])

                # print ("cta mirco f1: ", f1_score(cta_ts_true, cta_ts_pred, average="micro"))
                # print ("cta mirco f1 (col based): ", f1_score(cta_ts_true_col, cta_ts_pred_col, average="micro"))
                unique_labels = np.unique(cta_ts_true_col)
                unique_labels = [int(label) for label in unique_labels]
                unique_labels.sort()
                target_names = [data_dict[task]['idx2label'][label] for label in unique_labels]
                print(classification_report(
                    cta_ts_true_col,
                    cta_ts_pred_col,
                    digits=4,
                    labels=unique_labels,
                    target_names=target_names
                ))

                if write:
                    target_path = 'SOTAB_DATASET_FOLDER_PATH'
                    target_file = 'CTA_DBP_TEST_GT_CSV'
                    result_path = args.result_path
                    save_file = 'cta_dbp_test_{}.csv'.format(_iter)
                    with open(os.path.join(target_path, target_file), 'r') as fr:
                        with open(os.path.join(result_path, save_file), 'w') as fw:
                            for i, line in enumerate(fr.readlines()):
                                line = line.strip()
                                if len(line) == 0:
                                    continue
                                if i == 0:
                                    fw.write(line + ',label' + '\n')
                                    continue
                                tbname, col_idx = line.split(',')
                                predicted_label = test_dataset.idx2label[cta_result_dict[tbname[:-3]][int(col_idx)]['prediction']]
                                fw.write(tbname + ',' + col_idx + ',' + predicted_label + '\n')

            if 'CPA' in task:

                max_col = 9

                cpa_ts_true = []
                cpa_ts_pred = []
                cpa_ts_true_col = []
                cpa_ts_pred_col = []
                cpa_result_dict = {}

                for batch_i, (pred_list, tbnames, cls_idxs, cuts) in enumerate(zip(ts_pred_list, ts_tbnames, ts_cls_idx, ts_cuts)):

                    assert len(pred_list) == len(cls_idxs)
                    assert len(tbnames) == len(cuts)
                    combos1 = []
                    combos2 = []

                    for tbname, cut in zip(tbnames, cuts):
                        tdata = data_dict[task]['test'][tbname]
                        # tdata = data_dict[task]['validation'][tbname]

                        if cut == 0:
                            cut0 = sorted([(tbname, int(col[1]), l) for cnt, (col, l) in enumerate(zip(tdata['col_idx'], tdata['label'])) if cnt < max_col], key=lambda x: x[1])
                            combos1 += cut0
                        elif cut == 1:
                            cut1 = sorted([(tbname, int(col[1]), l) for cnt, (col, l) in enumerate(zip(tdata['col_idx'], tdata['label'])) if cnt >= max_col], key=lambda x: x[1])
                            combos1 += cut1
                        else:
                            raise ValueError("tables were split twice at most based on columns")

                    combos2 += [(cls_idx[0], p) for cls_idx, p in zip(cls_idxs, pred_list)]

                    assert len(combos1) == len(combos2)

                    for c1,c2 in zip(combos1, combos2):

                        cpa_ts_true.append(c1[2])
                        cpa_ts_pred.append(c2[1])

                        tbname = c1[0]
                        col_idx = c1[1]
                        predicted_label = c2[1]
                        true_label = c1[2]

                        if tbname not in cpa_result_dict:
                            cpa_result_dict[tbname] = {}

                        if col_idx in cpa_result_dict[tbname]:
                            cpa_result_dict[tbname][col_idx]['prediction'].append(predicted_label)
                        else:
                            cpa_result_dict[tbname][col_idx] = {}
                            cpa_result_dict[tbname][col_idx]['prediction'] = [predicted_label]
                            cpa_result_dict[tbname][col_idx]['label'] = true_label

                for tbname, tdata in cpa_result_dict.items():
                    for col_idx, item in tdata.items():
                        counter = Counter(cpa_result_dict[tbname][col_idx]['prediction'])
                        most_common_value = counter.most_common(1)[0][0]
                        cpa_result_dict[tbname][col_idx]['prediction'] = most_common_value

                        cpa_ts_true_col.append(cpa_result_dict[tbname][col_idx]['label'])
                        cpa_ts_pred_col.append(cpa_result_dict[tbname][col_idx]['prediction'])

                # print ("cpa mirco f1: ", f1_score(cpa_ts_true, cpa_ts_pred, average="micro"))
                # print(" cpa mirco f1 (col based): ", f1_score(cpa_ts_true_col, cpa_ts_pred_col, average="micro"))

                unique_labels = np.unique(cpa_ts_true_col)
                unique_labels = [int(label) for label in unique_labels]
                unique_labels.sort()
                target_names = [data_dict[task]['idx2label'][label] for label in unique_labels]
                print(classification_report(
                    cpa_ts_true_col,
                    cpa_ts_pred_col,
                    digits=4,
                    labels=unique_labels,
                    target_names=target_names
                ))

                if write:
                    target_path = 'SOTAB_DATASET_FOLDER_PATH'
                    target_file = 'CPA_DBP_TEST_GT_CSV'
                    result_path = args.result_path
                    save_file = 'cpa_dbp_test_{}.csv'.format(_iter)
                    with open(os.path.join(target_path, target_file), 'r') as fr:
                        with open(os.path.join(result_path, save_file), 'w') as fw:
                            for i, line in enumerate(fr.readlines()):
                                line = line.strip()
                                if len(line) == 0:
                                    continue
                                if i == 0:
                                    fw.write(line + ',label' +'\n')
                                    continue
                                tbname, col_idx1, col_idx2 = line.split(',')
                                predicted_label = test_dataset.idx2label[cpa_result_dict[tbname[:-3]][int(col_idx2)]['prediction']]
                                fw.write(tbname + ',' + col_idx1 + ',' + col_idx2 + ',' + predicted_label + '\n')

if __name__ == '__main__':
    inference()