import os
import json
import pprint
import argparse

import wandb
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import PairwiseDistance
from loss import TripletSoftMarginWithDistanceLoss
from negative import select_negative_multilabel
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from loaders.load_dataset import create_loader
from loaders.pt_graph import create_graph
from utils.data_utils import load_args_from_parser

from transformers import BertConfig
from layers.graph_nn import GCNNet
from layers.bert_task_layer import BertForMatching

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = load_args_from_parser()

def f1_score_multilabel(true_list, pred_list):
    conf_mat = multilabel_confusion_matrix(np.array(true_list),
                                           np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    p = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    r = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()

    micro_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] / conf_mat[:, 1, :].sum(axis=1)
    class_r = conf_mat[:, 1, 1] / conf_mat[:, :, 1].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    class_f1 = np.nan_to_num(class_f1)
    macro_f1 = class_f1.mean()
    return p, r, micro_f1, macro_f1, class_f1, conf_mat

def train():

    cta_training_mode = args.cta_training_mode
    cpa_training_mode = args.cpa_training_mode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    g, node_labels, edge_labels, added_label_ids, p_dict, p_positive_dict = create_graph(args)
    train_datasets, train_dataloaders, valid_datasets, valid_dataloaders, data_dict, task_num_class_dict = create_loader(args)

    batch_size = args.batch_size
    max_length = args.max_length
    num_train_epochs = args.epoch
    shortcut_name = args.shortcut_name
    note = args.note

    if args.from_scratch:
        tag_name = "model/{}-bs{}-ml-{}-{}".format("{}-fromscratch".format(shortcut_name), batch_size, max_length, note)
    else:
        tag_name = "model/{}-bs{}-ml-{}-{}".format(shortcut_name, batch_size, max_length, note)

    model_save_path = args.model_save_path
    save_path = os.path.join(model_save_path, tag_name)

    dirpath = os.path.dirname(save_path)
    if not os.path.exists(dirpath):
        print("{} not exists. Created".format(dirpath))
        os.makedirs(dirpath)

    wandb.login()
    wandb.init(project=args.wandb_project_name, dir=dirpath, config=args)

    if args.shortcut_name in [ "bert-base-multilingual-cased", "bert-base-uncased" ]:

        if args.from_scratch:
            # No pre-trained checkpoint
            model_config = BertConfig.from_pretrained(shortcut_name)
            model = BertForMatching(model_config)
        else:
            # Pre-trained checkpoint
            model = BertForMatching.from_pretrained(
                shortcut_name,
                output_attentions=False,
                output_hidden_states=False,
            )
    else:
        raise ("Could not find / load the language model: {}".format(args.shortcut_name))

    p_vocab_size = len(edge_labels['label2idx'])

    emb_dim = args.emb_dim
    gnn_params = {
        'emb_dim': emb_dim,
        'out_dim': emb_dim,
        'hidden_dim': emb_dim,
        'in_feat_dropout': 0.1,
        'dropout': 0.1,
        'L': args.L,
        'batch_norm': True,
        'residual': True,
        'p_vocab_size': p_vocab_size
    }

    model_g = GCNNet(g, gnn_params)
    model_g.to(device)

    g = g.to(device)

    model.init_by_task(task_num_class_dict, emb_dim)
    model.to(device)

    wandb.watch(model, log_freq=args.wandb_log_freq)

    optimizers = []
    schedulers = []
    loss_fns = []

    dist_func = PairwiseDistance(p=2)
    for i, (task, train_dataloader) in enumerate(zip(args.tasks, train_dataloaders)):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model_g.named_parameters()
                    if (not any(nd in n for nd in no_decay)) and ('node_unique_embeddings' in n)
                ],
                "weight_decay":
                    0.0,
                "lr": args.lr
            },
            {
                "params": [
                    p for n, p in model_g.named_parameters()
                    if (not any(nd in n for nd in no_decay)) and ('node_unique_embeddings' not in n)
                ],
                "weight_decay":
                    0.0,
                "lr": args.lr
            },
            {
                "params": [
                    p for n, p in model_g.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                    0.0,
                "lr": args.lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and n.split('.')[0] == 'bert'
                ],
                "weight_decay":
                    0.000001,
                "lr": args.lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and n.split('.')[0] == 'bert'
                ],
                "weight_decay":
                    0.0,
                "lr": args.lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and n.split('.')[0] != 'bert' and task in n
                ],
                "weight_decay":
                0.001,
                "lr": args.lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and n.split('.')[0] != 'bert' and task in n
                ],
                "weight_decay":
                    0.0,
                "lr": args.lr
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

        if 'CTA' in task:
            cta_t_total = len(train_dataloader) * num_train_epochs
            print('t total:', cta_t_total)
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0 if step < cta_t_total * 0.25 else  1.0 - (step - cta_t_total * 0.25) / (cta_t_total * (1 - 0.25)) )
            schedulers.append(scheduler)

        if 'CPA' in task:
            cpa_t_total = len(train_dataloader) * num_train_epochs
            print('t total:', cpa_t_total)
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0 if step < cpa_t_total * 0.25 else  1.0 - (step - cpa_t_total * 0.25) / (cpa_t_total * (1 - 0.25)) )
            schedulers.append(scheduler)

        optimizers.append(optimizer)
        loss_fns.append(
            TripletSoftMarginWithDistanceLoss(
                distance_function=dist_func,
                margin=1,
                swap=False,
                reduction='mean',
            )
        )

    # Best validation score could be zero
    best_vl_micro_f1s = [-1 for _ in range(len(args.tasks))]
    best_vl_macro_f1s = [-1 for _ in range(len(args.tasks))]
    loss_info_lists = [[] for _ in range(len(args.tasks))]

    for epoch in range(num_train_epochs):
        for k, (task, train_dataset, valid_dataset, train_dataloader,
                valid_dataloader, optimizer, scheduler, loss_fn,
                loss_info_list) in enumerate(
            zip(args.tasks, train_datasets, valid_datasets,
                train_dataloaders, valid_dataloaders, optimizers,
                schedulers, loss_fns, loss_info_lists)):
            t1 = time()

            if args.shortcut_name in ["bert-base-multilingual-cased", "bert-base-uncased"]:
                model.bert.pooler = model.poolers[task].to(device)
            else:
                raise ("Could not find the pooler.")

            model.train()
            tr_loss = 0.
            tr_pred_list = []
            tr_true_list = []

            vl_loss = 0.
            vl_pred_list = []
            vl_true_list = []

            train_dataset.generate_epoch()
            for batch_idx, batch in enumerate(train_dataloader):

                # LM
                logits, = model(
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                    task=task,
                )

                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(0)

                cls_indexes = batch['cls_idx']
                filtered_logits = torch.zeros(cls_indexes.shape[0], logits.shape[2]).to(device)

                assert cls_indexes.shape[0] == len(batch["label"])

                for n in range(cls_indexes.shape[0]):
                    # i -> table position in batch
                    # j -> column position in table
                    i, j = cls_indexes[n]
                    logit_n = logits[i, j, :]
                    filtered_logits[n] = logit_n

                # 3 layers -> batch -> multi labels -> multi edges
                # 2 layers (previous) -> batch -> multi labels
                if 'CPA' in task:
                    batch["flatten_label_l1"] = list()
                    batch["flatten_label_l2"] = list()
                    for property_labels, n in zip(batch["label"], batch["tbname"]):
                        batch["flatten_label_l1"].append([ r for m in property_labels for r in p_dict[m][n]]) # 3D the same content
                        for m in property_labels:
                            batch["flatten_label_l2"].extend(p_dict[m][n]) # 2D the same content

                anchor_origin = F.normalize(filtered_logits, p=2, dim=1)

                if 'CTA' in task:
                    anchor = torch.cat([
                        anchor_origin[pos].repeat(len(inner_list), 1) for pos, inner_list in enumerate(batch["label"])
                    ], dim=0)
                    positives_in_batch = [
                        inner_list for pos, inner_list in enumerate(batch["label"]) for _ in inner_list
                    ]
                else:
                    anchor = torch.cat([
                        anchor_origin[pos].repeat( len(inner_list), 1) for pos, inner_list in enumerate(batch["flatten_label_l1"])
                    ], dim=0)

                    positives_in_batch = [
                        inner_list for pos, inner_list in enumerate(batch["flatten_label_l1"]) for _ in inner_list
                    ]
                    assert len(positives_in_batch) == len(batch["flatten_label_l2"])
                assert len(positives_in_batch) == anchor.shape[0]
                label_origin = batch["label"]
                # flatten -> python list to tensor
                batch["label"] = torch.tensor([item for sublist in batch["label"] for item in sublist])

                # GNN
                num_all_nodes = len(node_labels['label2idx'])
                node_inputs = torch.tensor([idx for idx in range(num_all_nodes)], dtype=torch.int).to(device)
                edge_inputs = g.edata['rel_type']
                h, e = model_g._get_inputs(node_inputs, edge_inputs)
                _, _ = model_g(g, h, e)

                if 'CPA' in task:
                    batch_edge = torch.tensor(batch["flatten_label_l2"])
                    positive = model_g._get_outputs(batch_edge, task)
                else:
                    positive = model_g._get_outputs(batch["label"].cuda(), task)
                positive = F.normalize(positive, p=2, dim=1)

                if "CTA" in task:
                    vocab_size = len(node_labels['label2idx']) - len(added_label_ids['node'])
                elif "CPA" in task:
                    vocab_size = g.edata['rel_type'].shape[0] - len(added_label_ids['edge_gnn'])
                else:
                    raise ("Could not find the right task type.")

                vocab_inputs = torch.tensor([idx for idx in range(vocab_size)], dtype=torch.int).to(device)
                gnn_embeddings = model_g._get_outputs(vocab_inputs, task)
                gnn_embeddings = F.normalize(gnn_embeddings, p=2, dim=1)

                if 'CTA' in task:
                    negative, distances = select_negative_multilabel(cta_training_mode, batch_idx, anchor, positive, gnn_embeddings, positives_in_batch, dist_func)
                elif 'CPA' in task:
                    negative, distances = select_negative_multilabel(cpa_training_mode, batch_idx, anchor, positive, gnn_embeddings, positives_in_batch, dist_func)
                else:
                    raise ValueError('Could not find the training mode for the task.')
                loss = loss_fn(anchor, positive, negative)

                if 'CPA' in task:
                    distances2 = torch.square(torch.cdist(anchor_origin.unsqueeze(1), gnn_embeddings.unsqueeze(0), p=2).squeeze(1))
                    e2l = g.edata['rel_type'].cpu().detach().numpy().tolist()
                    for i, labels in enumerate(label_origin):
                        top_k_indices = torch.topk(distances2[i], len(labels), largest=False).indices  # Get indices of the top-k smallest distances
                        tr_pred_list.append([ e2l[e] for e in top_k_indices.cpu().detach().numpy().tolist()])
                        tr_true_list.append(labels)
                else:
                    distances2 = torch.square(torch.cdist(anchor_origin.unsqueeze(1), gnn_embeddings.unsqueeze(0), p=2).squeeze(1))
                    for i, labels in enumerate(label_origin):
                        top_k_indices = torch.topk(distances2[i], len(labels), largest=False).indices  # Get indices of the top-k smallest distances
                        tr_pred_list.append(top_k_indices.cpu().detach().numpy().tolist())
                        tr_true_list.append(labels)

                loss.backward()
                tr_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                model_g.zero_grad()

                if batch_idx % args.print_steps == 0:
                    print("{} Epoch {} - Step: {}; Current Loss: {:.10}; Total Average Loss: {:.10}".format(task,epoch + 1,batch_idx,loss.item(),tr_loss / (batch_idx + 1)))
                wandb.log({"{} Current Loss".format(task): loss.item(),
                           "{} Average Loss".format(task): tr_loss / (batch_idx + 1),
                           "{} Learning Rate".format(task): torch.tensor(scheduler.get_last_lr()[0])})

            tr_loss /= (len(train_dataset) / batch_size)

            mlb = MultiLabelBinarizer()
            mlb.fit_transform(tr_true_list)
            tr_true_list = mlb.transform(tr_true_list)
            tr_pred_list = mlb.transform(tr_pred_list)

            assert len(tr_true_list) == len(tr_pred_list)
            tr_micro_f1 = f1_score(tr_true_list, tr_pred_list, average="micro")
            tr_macro_f1 = f1_score(tr_true_list, tr_pred_list, average="macro")

            wandb.log(
                {"{} Training Macro F1".format(task): tr_macro_f1, "{} Training Micro F1".format(task): tr_micro_f1,
                 "{} Training Loss".format(task): tr_loss})

            # Validation
            model.eval()
            model_g.eval()
            valid_dataset.generate_epoch()
            for batch_idx, batch in enumerate(valid_dataloader):

                # LM
                logits, = model(
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                    task=task,
                )

                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(0)

                cls_indexes = batch['cls_idx']
                filtered_logits = torch.zeros(cls_indexes.shape[0], logits.shape[2]).to(device)

                for n in range(cls_indexes.shape[0]):
                    i, j = cls_indexes[n]
                    logit_n = logits[i, j, :]
                    filtered_logits[n] = logit_n

                if 'CPA' in task:
                    batch["flatten_label_l1"] = list()
                    batch["flatten_label_l2"] = list()
                    for property_labels, n in zip(batch["label"], batch["tbname"]):
                        batch["flatten_label_l1"].append([ r for m in property_labels for r in p_dict[m][n]]) # 3D the same content
                        for m in property_labels:
                            batch["flatten_label_l2"].extend(p_dict[m][n]) # 2D the same content

                if 'CTA' in task:
                    vocab_size = len(node_labels['label2idx']) - len(added_label_ids['node'])
                elif 'CPA' in task:
                    vocab_size = g.edata['rel_type'].shape[0] - len(added_label_ids['edge_gnn'])
                else:
                    raise (" Could not load the right taks type... ")

                anchor_origin = F.normalize(filtered_logits, p=2, dim=1)
                if 'CTA' in task:
                    anchor = torch.cat([
                        anchor_origin[pos].repeat(len(inner_list), 1) for pos, inner_list in enumerate(batch["label"])
                    ], dim=0)
                    positives_in_batch = [
                        inner_list for pos, inner_list in enumerate(batch["label"]) for _ in inner_list
                    ]
                else:
                    anchor = torch.cat([
                        anchor_origin[pos].repeat( len(inner_list), 1) for pos, inner_list in enumerate(batch["flatten_label_l1"])
                    ], dim=0)

                    positives_in_batch = [
                        inner_list for pos, inner_list in enumerate(batch["flatten_label_l1"]) for _ in inner_list
                    ]
                    assert len(positives_in_batch) == len(batch["flatten_label_l2"])

                assert len(positives_in_batch) == anchor.shape[0]
                label_origin = batch["label"]

                batch["label"] = torch.tensor([item for sublist in batch["label"] for item in sublist])

                # GNN
                num_all_nodes = len(node_labels['label2idx'])
                node_inputs = torch.tensor([idx for idx in range(num_all_nodes)], dtype=torch.int).to(device)
                edge_inputs = g.edata['rel_type']
                h, e = model_g._get_inputs(node_inputs, edge_inputs)
                _, _ = model_g(g, h, e)

                vocab_inputs = torch.tensor([idx for idx in range(vocab_size)], dtype=torch.int).to(device)
                gnn_embeddings = model_g._get_outputs(vocab_inputs, task)
                gnn_embeddings = F.normalize(gnn_embeddings, p=2, dim=1)

                distances2 = torch.cdist(anchor_origin.unsqueeze(1), gnn_embeddings.unsqueeze(0), p=2).squeeze(1)

                if 'CPA' in task:
                    e2l = g.edata['rel_type'].cpu().detach().numpy().tolist()
                    for i, labels in enumerate(label_origin):
                        top_k_indices = torch.topk(distances2[i], len(labels), largest=False).indices  # Get indices of the top-k smallest distances
                        vl_pred_list.append([ e2l[e] for e in top_k_indices.cpu().detach().numpy().tolist()])
                        vl_true_list.append(labels)
                else:
                    for i, labels in enumerate(label_origin):
                        top_k_indices = torch.topk(distances2[i], len(labels), largest=False).indices  # Get indices of the top-k smallest distances
                        vl_pred_list.append(top_k_indices.cpu().detach().numpy().tolist())
                        vl_true_list.append(labels)

                if 'CPA' in task:
                    batch_edge = torch.tensor(batch["flatten_label_l2"])
                    positive = model_g._get_outputs(batch_edge, task)
                else:
                    positive = model_g._get_outputs(batch["label"].cuda(), task)
                positive = F.normalize(positive, p=2, dim=1)

                # new semi-hard negative
                distances = torch.square(torch.cdist(anchor.unsqueeze(1), gnn_embeddings.unsqueeze(0), p=2).squeeze(1))
                p_distances = torch.square(dist_func(anchor, positive))
                p_distances_expanded = p_distances.view(-1, 1).expand(distances.shape[0], distances.shape[1])
                # Create a mask to exclude all positive distances
                mask = torch.ones_like(distances, dtype=torch.bool)  # Start with all distances as valid (True)
                for i, pos_indices in enumerate(positives_in_batch):
                    mask[i, pos_indices] = False  # Set the positive positions for each anchor to False
                # Apply semi-hard negative condition: distances greater than positive distances plus epsilon
                eps = 1e-04
                comparison_semi_hard = p_distances_expanded + eps < distances
                # Combine mask and semi-hard condition to get valid distances
                combined_mask = mask & comparison_semi_hard
                valid_distances = torch.where(combined_mask, distances, torch.max(distances))
                min_distances, min_indices = torch.min(valid_distances, dim=-1)
                negative = gnn_embeddings[min_indices]

                loss = loss_fn(anchor, positive, negative)
                vl_loss += loss.item()

            # test what labels are not there, not all the splits include all the labels.
            flat_vl_true_list = [item for sublist in vl_true_list for item in sublist]
            print([ l for l in range(255) if l not in set(flat_vl_true_list)])
            if 'CTA' in task:
                skip_classes = ['soccer.fifa', 'book.magazine', 'visual_art.visual_artist', 'chemistry.chemical_element', 'book.newspaper', 'aviation.airline', 'music.music_video_director']
            else:
                skip_classes = ['aviation.airline.hubs', 'tv.tv_character.appeared_in_tv_program-tv.regular_tv_appearance.actor', 'sports.sports_team.league-sports.sports_league_participation.from']

            target_names = [item[1] for item in sorted(
                [(key, value) for key, value in data_dict[task]['idx2label'].items() if value not in skip_classes],
                key=lambda x: int(x[0]))]
            total_classes = [item[0] for item in sorted(
                [(key, value) for key, value in data_dict[task]['idx2label'].items() if value not in skip_classes],
                key=lambda x: int(x[0]))]

            mlb = MultiLabelBinarizer(classes=total_classes)
            mlb.fit_transform(vl_true_list)
            vl_true_list = mlb.transform(vl_true_list)
            vl_pred_list = mlb.transform(vl_pred_list)

            vl_loss /= (len(valid_dataset) / batch_size)
            vl_p, vl_r, vl_micro_f1, vl_macro_f1, vl_class_f1, vl_conf_mat = f1_score_multilabel(vl_true_list, vl_pred_list)

            if vl_micro_f1 > best_vl_micro_f1s[k]:

                best_vl_micro_f1s[k] = vl_micro_f1

                negative_tag = 'cta{}cpa{}'.format(cta_training_mode, cpa_training_mode)

                model_filename = "{}={}_mirco_{}_{}.pt".format(tag_name, task, negative_tag, args.mark)
                save_path = os.path.join(model_save_path, model_filename)
                torch.save(model.state_dict(), save_path)

                model_g_filename = "{}={}_G_mirco_{}_{}.pt".format(tag_name, task, negative_tag, args.mark)
                save_g_path = os.path.join(model_save_path, model_g_filename)
                torch.save(model_g.state_dict(), save_g_path)

            if vl_macro_f1 > best_vl_macro_f1s[k]:

                best_vl_macro_f1s[k] = vl_macro_f1

                negative_tag = 'cta{}cpa{}'.format(cta_training_mode, cpa_training_mode)

                model_filename = "{}={}_marco_{}_{}.pt".format(tag_name, task, negative_tag, args.mark)
                save_path = os.path.join(model_save_path, model_filename)
                torch.save(model.state_dict(), save_path)

                model_g_filename = "{}={}_G_marco_{}_{}.pt".format(tag_name, task, negative_tag, args.mark)
                save_g_path = os.path.join(model_save_path, model_g_filename)
                torch.save(model_g.state_dict(), save_g_path)

            loss_info_list.append([tr_loss, tr_macro_f1, tr_micro_f1, vl_loss, vl_macro_f1, vl_micro_f1])

            t2 = time()
            print(
                "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f} "
                .format(epoch, task, tr_loss, tr_macro_f1, tr_micro_f1),
                "vl_loss={:.7f}  vl_p={:.4f}  vl_r={:.4f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
                .format(vl_loss, vl_p, vl_r, vl_macro_f1, vl_micro_f1, (t2 - t1)))

            print(classification_report(vl_true_list, vl_pred_list, digits=4, target_names=target_names))

            wandb.log({"{} Training Macro F1".format(task): tr_macro_f1, "{} Training Micro F1".format(task): tr_micro_f1, "{} Training Loss".format(task): tr_loss})
            wandb.log({"{} Evaluation Macro F1".format(task): vl_macro_f1, "{} Evaluation Micro F1".format(task): vl_micro_f1, "{} Evaluation Loss".format(task): vl_loss})

    for task, loss_info_list in zip(args.tasks, loss_info_lists):
        loss_info_df = pd.DataFrame(loss_info_list,
                                    columns=[
                                        "tr_loss", "tr_f1_macro_f1",
                                        "tr_f1_micro_f1", "vl_loss",
                                        "vl_f1_macro_f1", "vl_f1_micro_f1"
                                    ])
        if len(args.tasks) >= 2:
            loss_info_df.to_csv(os.path.join(model_save_path, "{}={}_loss_info.csv".format(tag_name, task)))
        else:
            loss_info_df.to_csv(os.path.join(model_save_path, "{}_loss_info.csv".format(tag_name)))

if __name__ == "__main__":
    train()
