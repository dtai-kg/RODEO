import sys
import json
import pickle
import random
import argparse

import numpy as np
import torch

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_json(path, dict):

    with open(path, 'w') as file:
        json.dump(dict, file, indent=4)

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

# We don't need to set the seed
# bc we want to generate different epoch datas
# check generate_epoch function in py_dataset.py
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# The effective priority: CLI args > JSON values > argparse defaults
def load_args_from_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tasks',
        action='append',
        help='tasks',
    )

    parser.add_argument(
        "--shortcut_name",
        default="bert-base-multilingual-cased",
        type=str,
        help="Huggingface model shortcut name ",
    )

    parser.add_argument(
        "--note",
        default="note",
        type=str,
        help="Note inside model tag",
    )

    parser.add_argument(
        "--max_col",
        default=10,
        type=int,
        help="Maximum number of columns to retain per table",
    )

    parser.add_argument(
        "--max_row",
        default=40,
        type=int,
        help="Maximum number of rows to retain per table",
    )

    parser.add_argument(
        "--max_cell_len",
        default=10,
        type=int,
        help="Maximum number of tokens per cell value",
    )

    parser.add_argument(
        "--num_row_per_sample",
        default=5,
        type=int,
        help="Number of rows per sub-table sample",
    )

    parser.add_argument(
        "--max_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size",
    )

    parser.add_argument(
        "--epoch",
        default=40,
        type=int,
        help="Number of epochs for training",
    )

    parser.add_argument(
        "--random_seed",
        default=4649,
        type=int,
        help="Random seed",
    )

    # Not using
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.,
        help="Warmup ratio",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )

    parser.add_argument(
        "--decay",
        type=float,
        default=5e-5,
        help="decay",
    )

    parser.add_argument(
        "--from_scratch",
        action="store_true",
        default=True,
        help="Training from scratch",
    )

    parser.add_argument(
        "--single_subtable_on",
        action="store_true",
        default=False,
        help="use one subtable per table",
    )

    parser.add_argument(
        "--wandb_project_name",
        default="RODEO_PROJECT",
        type=str,
        help="Weights & Biases project name for experiment tracking",
    )

    parser.add_argument(
        "--wandb_log_freq",
        default=100,
        type=int,
        help="Log metrics to W&B every N steps",
    )

    parser.add_argument(
        "--print_steps",
        default=100,
        type=int,
        help="Print training progress every N steps",
    )

    parser.add_argument(
        "--emb_dim",
        default=64,
        type=int,
        help="GNN embedding dimension",
    )

    parser.add_argument(
        "--L",
        default=3,
        type=int,
        help="Number of GNN layers",
    )

    parser.add_argument(
        "--mark",
        default="200",
        type=str,
        help="Checkpoint tag for saved model filenames",
    )

    parser.add_argument(
        "--result_path",
        default="./results",
        type=str,
        help="Directory for inference output CSVs",
    )

    parser.add_argument(
        "--cta_training_mode",
        default="1_1",
        type=str,
        help="Negative ratio for CTA (e.g. 1_1)",
    )

    parser.add_argument(
        "--cpa_training_mode",
        default="1_2",
        type=str,
        help="Negative ratio for CPA (e.g. 1_2)",
    )

    parser.add_argument(
        "--encoding",
        default="base",
        type=str,
        choices=["base", "llm"],
        help="Tokenization variant: 'base' for BERT, 'llm' for Qwen",
    )

    parser.add_argument(
        "--sotab_dataset_path",
        default="/apollo/users/dya/dataset/semtab",
        type=str,
        help="Root directory of the SOTAB dataset",
    )

    parser.add_argument(
        "--turl_dataset_path",
        default="/apollo/users/dya/dataset/wikitables",
        type=str,
        help="Root directory of the TURL (WikiTables) dataset",
    )

    parser.add_argument(
        "--gittab_dataset_path",
        default="/apollo/users/dya/dataset/gittable_numeric",
        type=str,
        help="Root directory of the GitTables dataset",
    )

    parser.add_argument(
        "--model_save_path",
        default="MODEL_SAVE_PATH",
        type=str,
        help="Directory for saving model checkpoints",
    )

    parser.add_argument(
        '--load_json',
        default=None,
        type=str,
        help='Load settings from file in json format. Command line options override values in file.'
    )

    args = parser.parse_args()

    if args.load_json:
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    return args