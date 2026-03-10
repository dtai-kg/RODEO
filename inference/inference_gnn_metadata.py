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

from layers.graph_nn import GCNNet
from train.loss import TripletSoftMarginWithDistanceLoss
from utils.data_utils import load_args_from_parser
from loaders.load_dataset import create_loader, create_test_loader
from loaders.pt_graph import create_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = load_args_from_parser()

def inference():

    # hard coded -> line 78,79 ->  TODO -> update model names

    # iterations: large tables with massive rows and columns => random splitting may lead to different results.
    iteration = 3

    g, node_labels, edge_labels, added_label_ids, p_dict_by_topic = create_graph(args)
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
            model_filename = "{}={}_marco_cta1_1cpa1_2_100.pt".format(tag_name, task)
            model_g_filename = "{}={}_G_marco_cta1_1cpa1_2_100.pt".format(tag_name, task)
            print (model_filename)
            print (model_g_filename)

            load_path = os.path.join(args.model_save_path, model_filename)
            sd = torch.load(load_path, map_location=device)
            model.load_state_dict(sd)

        else:
            raise ("Could not find / load the language model: {}".format(args.shortcut_name))

        gnn_params = {
            'emb_dim': emb_dim,
            'out_dim': emb_dim,
            'hidden_dim': emb_dim,
            'in_feat_dropout': 0.1,
            'dropout': 0.1,
            'L': args.L,
            'batch_norm': True,
            'residual': True,
            'p_vocab_size': len(edge_labels['label2idx'])
        }

        model_g = GCNNet(g, gnn_params)
        model_g.to(device)

        load_path = os.path.join(args.model_save_path, model_g_filename)
        sd_g = torch.load(load_path, map_location=device)
        model_g.load_state_dict(sd_g)

        g = g.to(device)

        model.eval()
        model_g.eval()

        for iter in range(iteration):

            ts_pred_list = []
            ts_tbnames = []
            ts_col_nos = []
            ts_cls_idx = []
            ts_cuts = []

            test_dataset.generate_epoch()
            print("Test data Length: ", len(test_dataset))
            for batch_idx, batch in enumerate(test_dataloader):

                # table
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

                # graph
                num_all_nodes = len(node_labels['label2idx'])
                node_inputs = torch.tensor([idx for idx in range(num_all_nodes)], dtype=torch.int).to(device)
                edge_inputs = g.edata['rel_type']
                h, e = model_g._get_inputs(node_inputs, edge_inputs)
                _, _ = model_g(g, h, e)
                if "CTA" in task:
                    vocab_size = len(node_labels['label2idx']) - len(added_label_ids['node'])
                elif "CPA" in task:
                    vocab_size = g.edata['rel_type'].shape[0] - len(added_label_ids['edge_gnn'])
                else:
                    raise ("Could not find the right task type.")
                vocab_inputs = torch.tensor([idx for idx in range(vocab_size)], dtype=torch.int).to(device)

                gnn_embeddings = model_g._get_outputs(vocab_inputs, task)
                gnn_embeddings = F.normalize(gnn_embeddings, p=2, dim=1)

                # distances
                distances = torch.square(torch.cdist(anchor.unsqueeze(1), gnn_embeddings.unsqueeze(0), p=2).squeeze(1))

                # all the results
                ts_cls_idx.append(copy.deepcopy(cls_indexes.cpu().detach().numpy().tolist()))
                if 'CTA' in task:
                    ts_pred_list.append(distances.argmin(-1).cpu().detach().numpy().flatten().tolist())
                else:
                    e2l = g.edata['rel_type'].cpu().detach().numpy().tolist()
                    ts_pred_list.append([ e2l[e] for e in distances.argmin(-1).cpu().detach().numpy().flatten().tolist()])

                ts_tbnames.append(batch["tbname"])
                ts_col_nos.append(batch["col_nos"])
                ts_cuts.append(batch["cut"])

            if 'CPA' in task:
                continue

            # majority voting with subtables
            if 'CTA' in task:

                max_col = 20

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

                        cut1 = sorted([(tbname, int(col), l, category) for cnt, (col, l, category) in enumerate(zip(tdata['col_idx'], tdata['label'], tdata['category'])) if (cnt >= cut * max_col) and (cnt < (cut+1) * max_col)], key=lambda x: x[1])
                        combos1 += cut1

                    combos2 += [(cls_idx[0], p) for cls_idx, p in zip(cls_idxs, pred_list)]

                    assert len(combos1) == len(combos2)

                    for c1,c2 in zip(combos1, combos2):

                        cta_ts_true.append(c1[2])
                        cta_ts_pred.append(c2[1])

                        tbname = c1[0]
                        col_idx = c1[1]
                        predicted_label = c2[1]
                        true_label = c1[2]
                        category = c1[3]

                        if tbname not in cta_result_dict:
                            cta_result_dict[tbname] = {}

                        if col_idx in cta_result_dict[tbname]:
                            cta_result_dict[tbname][col_idx]['prediction'].append(predicted_label)
                        else:
                            cta_result_dict[tbname][col_idx] = {'category': category}
                            cta_result_dict[tbname][col_idx]['prediction'] = [predicted_label]
                            cta_result_dict[tbname][col_idx]['label'] = true_label

                for tbname, tdata in cta_result_dict.items():
                    for col_idx, item in tdata.items():
                        counter = Counter(cta_result_dict[tbname][col_idx]['prediction'])
                        most_common_value = counter.most_common(1)[0][0]
                        cta_result_dict[tbname][col_idx]['prediction'] = most_common_value

                        assert cta_result_dict[tbname][col_idx]['category'] in ["numercial", "textual"]

                        # if cta_result_dict[tbname][col_idx]['category'] == "textual":
                        # if cta_result_dict[tbname][col_idx]['category'] == "numercial":
                        #     cta_ts_true_col.append(cta_result_dict[tbname][col_idx]['label'])
                        #     cta_ts_pred_col.append(cta_result_dict[tbname][col_idx]['prediction'])

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

if __name__ == '__main__':
    inference()


