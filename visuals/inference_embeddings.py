import os
import sys
import copy

import wandb

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import load_args_from_parser
from loaders.load_dataset import create_test_loader
from loaders.pt_graph import create_graph
from layers.graph_nn import GCNNet
from utils.data_utils import save

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = load_args_from_parser()

def inference():

    g, node_labels, edge_labels, added_label_ids, p_dict_by_topic = create_graph(args)
    valid_datasets, valid_dataloaders, test_datasets, test_dataloaders, data_dict, task_num_class_dict = create_test_loader(args)

    note = args.note
    shortcut_name = args.shortcut_name
    batch_size = args.batch_size
    max_length = args.max_length
    emb_dim = args.emb_dim

    tag_name = "model/{}-bs{}-ml-{}-{}".format(shortcut_name, batch_size, max_length, note)

    for k, (task, valid_dataset, test_dataset, valid_dataloader,
            test_dataloader) in enumerate(
                zip(args.tasks, valid_datasets, test_datasets,
                    valid_dataloaders, test_dataloaders)):

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
            model_g_filename = "{}={}_G_marco_cta1_1cpa1_2_200.pt".format(tag_name, task)
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

        # CTA only
        if 'CPA' in task:
            continue

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

            num_all_nodes = len(node_labels['label2idx'])
            node_inputs = torch.tensor([idx for idx in range(num_all_nodes)], dtype=torch.int).to(device)
            edge_inputs = g.edata['rel_type']
            h, e = model_g._get_inputs(node_inputs, edge_inputs)
            _, _ = model_g(g, h, e)

            if len(logits.shape) == 2:
                logits = logits.unsqueeze(0)

            cls_indexes = batch['cls_idx']
            filtered_logits = torch.zeros(cls_indexes.shape[0], logits.shape[2]).to(device)

            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                logit_n = logits[i, j, :]
                filtered_logits[n] = logit_n

            anchor = F.normalize(filtered_logits, p=2, dim=1)

            # CTA only
            vocab_size = len(node_labels['label2idx']) - len(added_label_ids['node'])

            vocab_inputs = torch.tensor([idx for idx in range(vocab_size)], dtype=torch.int).to(device)
            gnn_embeddings = model_g._get_outputs(vocab_inputs, task)
            gnn_embeddings = F.normalize(gnn_embeddings, p=2, dim=1)

            ts_cls_idx.append(copy.deepcopy(cls_indexes.cpu().detach().numpy().tolist()))
            ts_pred_list.append(anchor.cpu().detach().numpy().tolist())

            ts_tbnames.append(batch["tbname"])
            ts_col_nos.append(batch["col_nos"])
            ts_cuts.append(batch["cut"])

        gnn_embeddings = gnn_embeddings.cpu().detach().numpy().tolist()

        max_col = 10
        cta_ts_true_col = []
        cta_ts_pred_col = []

        for batch_i, (pred_list, tbnames, cls_idxs, cuts) in enumerate(zip(ts_pred_list, ts_tbnames, ts_cls_idx, ts_cuts)):

            assert len(pred_list) == len(cls_idxs)
            assert len(tbnames) == len(cuts)
            combos1 = []
            combos2 = []

            for tbname, cut in zip(tbnames, cuts):

                tdata = data_dict[task]['test'][tbname]

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

                true_label = c1[2]
                predicted_label = c2[1]

                if c1[2] in [29, 31, 16, 24, 35, 63, 36, 45, 22, 5, 80, 8, 79, 50, 30, 26]:
                    cta_ts_true_col.append(true_label)
                    cta_ts_pred_col.append(predicted_label)

                assert len(predicted_label) == 64, print(len(predicted_label))

        subj_class = ['Product/name', 'Recipe/name', 'MusicAlbum/name', 'Event/name', 'LocalBusiness/name',
         'TVEpisode/name', 'Hotel/name', 'Restaurant/name', 'MusicRecording/name', 'Person/name',
        'JobPosting/name', 'Place/name', 'Museum/name', 'SportsEvent/name', 'Movie/name', 'Book/name']
        subj_class_idxs = [29, 31, 16, 24, 35, 63, 36, 45, 22, 5, 80, 8, 79, 50, 30, 26]

        save('cta_embeddings.pkl', [gnn_embeddings, cta_ts_pred_col, cta_ts_true_col, subj_class_idxs, subj_class])

if __name__ == '__main__':
    inference()


