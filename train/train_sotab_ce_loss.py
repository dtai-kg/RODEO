import os
import argparse
import json
from time import time
import wandb

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

import torch
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from utils.data_utils import load_args_from_parser
from loaders.load_dataset import create_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = load_args_from_parser()

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_datasets, train_dataloaders, valid_datasets, valid_dataloaders, data_dict, task_num_class_dict = create_loader(args)

    batch_size = args.batch_size
    max_length = args.max_length
    num_train_epochs = args.epoch
    warm_up_ratio = args.warmup

    shortcut_name = args.shortcut_name
    note = args.note

    if args.from_scratch:
        tag_name = "model/{}-bs{}-ml-{}-{}".format("{}-fromscratch".format(shortcut_name), batch_size, max_length, note)
    else:
        tag_name = "model/{}-bs{}-ml-{}-{}".format(shortcut_name, batch_size, max_length, note)

    print(tag_name)

    model_save_path = args.model_save_path
    save_path = os.path.join(model_save_path, tag_name)

    dirpath = os.path.dirname(save_path)
    if not os.path.exists(dirpath):
        print("{} not exists. Created".format(dirpath))
        os.makedirs(dirpath)

    wandb.login()
    wandb.init(project=args.wandb_project_name, dir=dirpath, config=args)

    if args.shortcut_name in [ "bert-base-uncased", "bert-base-multilingual-cased" ]:

        from transformers import BertConfig
        from layers.bert_task_layer import BertForMultiOutputClassification

        if args.from_scratch:
            # No pre-trained checkpoint
            model_config = BertConfig.from_pretrained(shortcut_name)
            model = BertForMultiOutputClassification(model_config)
        else:
            # Pre-trained checkpoint
            model = BertForMultiOutputClassification.from_pretrained(
                shortcut_name,
                output_attentions=False,
                output_hidden_states=False,
            )
    else:
        raise ("Could not find / load the language model: {}".format(args.shortcut_name))

    model.init_by_task(task_num_class_dict)
    model.to(device)

    wandb.watch(model, log_freq=args.wandb_log_freq)

    optimizers = []
    schedulers = []
    loss_fns = []

    for i, (task, train_dataloader) in enumerate(zip(args.tasks, train_dataloaders)):
        t_total = len(train_dataloader) * num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and
                       n.split('.')[0] == 'bert'
                ],
                "weight_decay":
                    0.0001
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and
                       n.split('.')[0] == 'bert'
                ],
                "weight_decay":
                    0.0
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and n.split('.')[0] != 'bert' and task in n
                ],
                "weight_decay":
                    0.01 if 'CPA' in task else 0.001
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and n.split('.')[0] != 'bert' and task in n
                ],
                "weight_decay":
                    0.0
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

        warmup_steps = int(t_total * warm_up_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

        loss_fns.append(CrossEntropyLoss())

    # Best validation score could be zero
    best_vl_micro_f1s = [-1 for _ in range(len(args.tasks))]
    loss_info_lists = [[] for _ in range(len(args.tasks))]

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

                tr_pred_list += filtered_logits.argmax(1).cpu().detach().numpy().tolist()
                tr_true_list += batch["label"].cpu().detach().numpy().tolist()
                loss = loss_fn(filtered_logits, batch["label"].cuda())

                loss.backward()
                tr_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                if batch_idx % args.print_steps == 0:
                    print("{} Epoch {} - Step: {}; Current Loss: {:.10}; Total Average Loss: {:.10}".format(task,epoch + 1,batch_idx,loss.item(),tr_loss / (batch_idx + 1)))
                wandb.log({"{} Current Loss".format(task): loss.item(),
                           "{} Average Loss".format(task): tr_loss / (batch_idx + 1),
                           "{} Learning Rate".format(task): torch.tensor(scheduler.get_last_lr()[0])})

            tr_loss /= (len(train_dataset) / batch_size)

            tr_micro_f1 = f1_score(tr_true_list, tr_pred_list, average="micro")
            tr_macro_f1 = f1_score(tr_true_list, tr_pred_list, average="macro")

            # Validation
            model.eval()
            valid_dataset.generate_epoch()
            for batch_idx, batch in enumerate(valid_dataloader):

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

                vl_pred_list += filtered_logits.argmax(1).cpu().detach().numpy().tolist()
                vl_true_list += batch["label"].cpu().detach().numpy().tolist()

                loss = loss_fn(filtered_logits, batch["label"].cuda())
                vl_loss += loss.item()

            vl_loss /= (len(valid_dataset) / batch_size)
            vl_micro_f1 = f1_score(vl_true_list, vl_pred_list, average="micro")
            vl_macro_f1 = f1_score(vl_true_list, vl_pred_list, average="macro")

            if vl_micro_f1 > best_vl_micro_f1s[k]:
                best_vl_micro_f1s[k] = vl_micro_f1
                model_filename = "{}={}_best_micro_f1.pt".format(tag_name, task)
                save_path = os.path.join(model_save_path, model_filename)
                torch.save(model.state_dict(), save_path)

            loss_info_list.append([tr_loss, tr_macro_f1, tr_micro_f1, vl_loss, vl_macro_f1, vl_micro_f1])

            t2 = time()
            print(
                "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f} "
                .format(epoch, task, tr_loss, tr_macro_f1, tr_micro_f1),
                "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
                .format(vl_loss, vl_macro_f1, vl_micro_f1, (t2 - t1)))
            print(classification_report(vl_true_list, vl_pred_list, digits = 4, target_names= [ item[1] for item in sorted([(key, value) for key, value in data_dict[task]['idx2label'].items()], key=lambda x: int(x[0])) ]  ))

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
