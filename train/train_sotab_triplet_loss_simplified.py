import os
import json
import argparse
import wandb
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import PairwiseDistance
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from layers.graph_nn import SemanticLabelEncoder
from loss import TripletSoftMarginWithDistanceLoss
from negative import select_negative

from loaders.load_dataset import create_loader
from utils.data_utils import load_args_from_parser

# load settings from json file
args = load_args_from_parser()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():

    # determine negative ratio mode
    cta_training_mode = args.cta_training_mode
    cpa_training_mode = args.cpa_training_mode
    dist_func = PairwiseDistance(p=2)

    # load task datasets
    train_datasets, train_dataloaders, valid_datasets, valid_dataloaders, data_dict, task_num_class_dict = create_loader(args)

    # load configurations
    batch_size = args.batch_size
    max_length = args.max_length
    num_train_epochs = args.epoch
    shortcut_name = args.shortcut_name
    note = args.note

    # model path
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

    # wandb tracking
    wandb.login()
    wandb.init(project=args.wandb_project_name, dir=dirpath, config=args)

    # load language model
    if args.shortcut_name in [ "bert-base-uncased", "bert-base-multilingual-cased" ]:

        from transformers import BertConfig
        from layers.bert_task_layer import BertForMatching

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

    num_unique_nodes = next(task_num_class_dict[task] for task in args.tasks if 'CTA' in task)
    num_unique_edges = next(task_num_class_dict[task] for task in args.tasks if 'CPA' in task)

    emb_dim = args.emb_dim

    model_l = SemanticLabelEncoder({
        'emb_dim': args.emb_dim,
        'num_unique_nodes': num_unique_nodes,
        'num_unique_edges': num_unique_edges
    })
    model_l.to(device)

    model.init_by_task(task_num_class_dict, emb_dim)
    model.to(device)

    wandb.watch(model, log_freq=args.wandb_log_freq)

    # load optimizers / schedulers / losses for multi-task learning
    optimizers = []
    schedulers = []
    loss_fns = []

    for i, (task, train_dataloader) in enumerate(zip(args.tasks, train_dataloaders)):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model_l.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                    0.001,
                "lr": args.lr
            },
            {
                "params": [
                    p for n, p in model_l.named_parameters()
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
                    0.0001,
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
        scheduler = None
        if 'CTA' in task:
            cta_t_total = len(train_dataloader) * num_train_epochs
            print('t total:', cta_t_total)
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0 if step < cta_t_total * 0.25 else  1.0 - (step - cta_t_total * 0.25) / (cta_t_total * (1 - 0.25)) )

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

    # initialize metrics
    best_vl_micro_f1s = [-1 for _ in range(len(args.tasks))]
    best_vl_macro_f1s = [-1 for _ in range(len(args.tasks))]
    loss_info_lists = [[] for _ in range(len(args.tasks))]

    # training
    for epoch in range(num_train_epochs):
        for k, (task, train_dataset, valid_dataset, train_dataloader,
                valid_dataloader, optimizer, scheduler, loss_fn,
                loss_info_list) in enumerate(
            zip(args.tasks, train_datasets, valid_datasets,
                train_dataloaders, valid_dataloaders, optimizers,
                schedulers, loss_fns, loss_info_lists)):
            t1 = time()

            if args.shortcut_name in ["bert-base-uncased", "bert-base-multilingual-cased"]:
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

                # run language model -> anchor
                logits, = model(
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                    task=task,
                )

                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(0)

                cls_indexes = batch['cls_idx']
                filtered_logits = torch.zeros(cls_indexes.shape[0], logits.shape[2]).to(device)

                assert cls_indexes.shape[0] == batch["label"].shape[0]

                for n in range(cls_indexes.shape[0]):
                    # i -> table position in batch
                    # j -> column position in table
                    i, j = cls_indexes[n]
                    logit_n = logits[i, j, :]
                    filtered_logits[n] = logit_n

                anchor = F.normalize(filtered_logits, p=2, dim=1)

                positive = model_l._get_outputs(batch["label"].cuda(), task)
                positive = F.normalize(positive, p=2, dim=1)

                vocab_size = task_num_class_dict[task]
                vocab_inputs = torch.tensor([idx for idx in range(vocab_size)], dtype=torch.int).to(device)
                label_embeddings = model_l._get_outputs(vocab_inputs, task)
                label_embeddings = F.normalize(label_embeddings, p=2, dim=1)

                distances = torch.square(torch.cdist(anchor.unsqueeze(1), label_embeddings.unsqueeze(0), p=2).squeeze(1))
                p_distances = torch.square(dist_func(anchor, positive))
                p_distances_expanded = p_distances.view(-1, 1).expand(distances.shape[0], distances.shape[1])

                if 'CTA' in task:
                    negative = select_negative(cta_training_mode, batch_idx, distances, p_distances_expanded, label_embeddings)
                elif 'CPA' in task:
                    negative = select_negative(cpa_training_mode, batch_idx, distances, p_distances_expanded, label_embeddings)
                else:
                    raise ValueError('Could not find the training mode for the task.')

                # compute train loss
                loss = loss_fn(anchor, positive, negative)

                tr_pred_list += distances.argmin(-1).cpu().detach().numpy().flatten().tolist()
                tr_true_list += batch["label"].cpu().detach().numpy().tolist()

                loss.backward()
                tr_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                model_l.zero_grad()

                if batch_idx % args.print_steps == 0:
                    print("{} Epoch {} - Step: {}; Current Loss: {:.10}; Total Average Loss: {:.10}".format(task,epoch + 1,batch_idx,loss.item(),tr_loss / (batch_idx + 1)))
                wandb.log({"{} Current Loss".format(task): loss.item(),
                           "{} Average Loss".format(task): tr_loss / (batch_idx + 1),
                           "{} Learning Rate".format(task): torch.tensor(scheduler.get_last_lr()[0])})

            tr_loss /= (len(train_dataset) / batch_size)

            assert len(tr_true_list) == len(tr_pred_list)
            tr_micro_f1 = f1_score(tr_true_list, tr_pred_list, average="micro")
            tr_macro_f1 = f1_score(tr_true_list, tr_pred_list, average="macro")

            wandb.log(
                {"{} Training Macro F1".format(task): tr_macro_f1, "{} Training Micro F1".format(task): tr_micro_f1,
                 "{} Training Loss".format(task): tr_loss})

            # Validation
            model.eval()
            model_l.eval()
            valid_dataset.generate_epoch()
            for batch_idx, batch in enumerate(valid_dataloader):

                # run language model -> anchor
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

                anchor = F.normalize(filtered_logits, p=2, dim=1)

                vocab_size = task_num_class_dict[task]
                vocab_inputs = torch.tensor([idx for idx in range(vocab_size)], dtype=torch.int).to(device)
                label_embeddings = model_l._get_outputs(vocab_inputs, task)
                label_embeddings = F.normalize(label_embeddings, p=2, dim=1)

                # prediction (the shortest distance between anchor and all the GNN embeddings)
                distances = torch.cdist(anchor.unsqueeze(1), label_embeddings.unsqueeze(0), p=2).squeeze(1)
                vl_pred_list += distances.argmin(-1).cpu().detach().numpy().flatten().tolist()

                positive = model_l._get_outputs(batch["label"].cuda(), task)
                positive = F.normalize(positive, p=2, dim=1)

                # get GNN outputs + anchor -> negative (semi-hard)
                p_distances = torch.square(dist_func(anchor, positive))
                p_distances_expanded = p_distances.view(-1, 1).expand(distances.shape[0], distances.shape[1])
                eps = 1e-04
                comparison = p_distances_expanded + eps < distances
                valid_distances = torch.where(comparison, distances, torch.max(distances))
                min_distances, min_indices = torch.min(valid_distances, dim=-1)
                negative = label_embeddings[min_indices]

                vl_true_list += batch["label"].cpu().detach().numpy().tolist()

                # compute val loss
                loss = loss_fn(anchor, positive, negative)
                vl_loss += loss.item()

            vl_loss /= (len(valid_dataset) / batch_size)
            vl_micro_f1 = f1_score(vl_true_list, vl_pred_list, average="micro")
            vl_macro_f1 = f1_score(vl_true_list, vl_pred_list, average="macro")

            # saving models based on the best F1 scores
            if vl_micro_f1 > best_vl_micro_f1s[k]:

                best_vl_micro_f1s[k] = vl_micro_f1

                negative_tag = 'cta{}cpa{}'.format(cta_training_mode, cpa_training_mode)

                model_filename = "{}={}_mirco_{}_{}.pt".format(tag_name, task, negative_tag, args.mark)
                save_path = os.path.join(model_save_path, model_filename)
                torch.save(model.state_dict(), save_path)

                model_l_filename = "{}={}_L_mirco_{}_{}.pt".format(tag_name, task, negative_tag, args.mark)
                save_g_path = os.path.join(model_save_path, model_l_filename)
                torch.save(model_l.state_dict(), save_g_path)

            if vl_macro_f1 > best_vl_macro_f1s[k]:

                best_vl_macro_f1s[k] = vl_macro_f1

                negative_tag = 'cta{}cpa{}'.format(cta_training_mode, cpa_training_mode)

                model_filename = "{}={}_marco_{}_{}.pt".format(tag_name, task, negative_tag, args.mark)
                save_path = os.path.join(model_save_path, model_filename)
                torch.save(model.state_dict(), save_path)

                model_l_filename = "{}={}_L_marco_{}_{}.pt".format(tag_name, task, negative_tag, args.mark)
                save_g_path = os.path.join(model_save_path, model_l_filename)
                torch.save(model_l.state_dict(), save_g_path)

            loss_info_list.append([tr_loss, tr_macro_f1, tr_micro_f1, vl_loss, vl_macro_f1, vl_micro_f1])

            t2 = time()
            print(
                "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f} "
                .format(epoch, task, tr_loss, tr_macro_f1, tr_micro_f1),
                "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
                .format(vl_loss, vl_macro_f1, vl_micro_f1, (t2 - t1)))

            print(classification_report(vl_true_list, vl_pred_list, digits=4, target_names=[item[1] for item in sorted([(key, value) for key, value in data_dict[task]['idx2label'].items()], key=lambda x: int(x[0]))]))

            wandb.log({"{} Training Macro F1".format(task): tr_macro_f1, "{} Training Micro F1".format(task): tr_micro_f1, "{} Training Loss".format(task): tr_loss})
            wandb.log({"{} Evaluation Macro F1".format(task): vl_macro_f1, "{} Evaluation Micro F1".format(task): vl_micro_f1, "{} Evaluation Loss".format(task): vl_loss})

if __name__ == "__main__":
    train()
