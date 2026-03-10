import os
import csv
import json
import neo4j
import pickle
import pprint
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from neo4j import GraphDatabase
from scipy.cluster.hierarchy import linkage, leaves_list

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
from scipy.sparse import hstack

from utils.file_registry import get_path

def common_substrings(strings, min_count=4, min_fraction=0.5, min_length=3):

    if not strings:
        return []

    n = len(strings)
    threshold = max(min_count, int(np.ceil(n * min_fraction)))

    first = strings[0]
    substring_counter = {}

    substrings = set()
    for i in range(len(first)):
        for j in range(i + 1, len(first) + 1):
            substr = first[i:j]
            if len(substr) >= min_length:
                substrings.add(substr)

    for substr in substrings:
        count = sum(substr in s for s in strings)
        if count >= threshold:
            substring_counter[substr] = count

    if not substring_counter:
        return []

    sorted_substrings = sorted(substring_counter.keys(), key=lambda s: (-len(s), s))

    final_substrings = []
    for s in sorted_substrings:
        if any(s in kept for kept in final_substrings):
            continue
        final_substrings.append(s)

    return final_substrings

def cluster_table_combined_with_ranking(file_to_labels, distance_threshold=1.5):

    # print(f"Total files loaded: {len(file_to_labels)}")

    filenames = list(file_to_labels.keys())
    label_strings = []
    filename_strings = []
    cleaned_label_map = {}

    for fname in filenames:
        labels = file_to_labels[fname]

        cleaned_labels = [re.sub(r'[\d_\-]+', ' ', label[2:].lower()).strip() for label in labels]
        cleaned_labels = list(set(cleaned_labels))

        cleaned_label_map[fname] = set(cleaned_labels)

        # print(set(cleaned_labels))

        pairs = ["{}_{}".format(*sorted(pair)) for pair in combinations(cleaned_labels, 2)]
        combined_features = cleaned_labels + pairs
        label_text = " ".join(combined_features)
        label_strings.append(label_text)

        name = re.sub(r'\.parquet$', '', fname)
        after = re.search(r'_licensed_(.*)', name)
        before = re.match(r'(.*)_tables_licensed_', name)
        after_part = after.group(1) if after else ''
        before_part = before.group(1) if before else ''
        after_part = re.sub(r'[\d_\-]+', ' ', after_part.lower()).strip()
        before_part = re.sub(r'[\d_\-]+', ' ', before_part.lower()).strip()
        filename_text = after_part if after_part else before_part
        filename_strings.append(filename_text)

    label_vectorizer = TfidfVectorizer()
    X_labels = label_vectorizer.fit_transform(label_strings)

    filename_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    X_filenames = filename_vectorizer.fit_transform(filename_strings)

    X_combined = hstack([X_labels * 1.0, X_filenames * 0.5])

    clustering = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None, linkage='ward')
    cluster_labels = clustering.fit_predict(X_combined.toarray())

    clusters = defaultdict(list)
    for cluster_id, fname in zip(cluster_labels, filenames):
        clusters[cluster_id].append(fname)

    # print(f"\nTotal clusters found: {len(clusters)}")

    cluster_stats = []

    max0_count = 0
    max1_count = 0
    avg_le1_count = 0
    median0_count = 0
    median_le1_count = 0

    max0_files = 0
    max1_files = 0
    avg_le1_files = 0
    median0_files = 0
    median_le1_files = 0

    for cluster_id, files in clusters.items():
        if len(files) == 1:
            cluster_stats.append({
                "cluster_id": cluster_id,
                "files": files,
                "max_diff": None,
            })
            continue

        diffs = []
        for f1, f2 in combinations(files, 2):
            diff_count = len(cleaned_label_map[f1].symmetric_difference(cleaned_label_map[f2]))
            diffs.append(diff_count)

        max_diff = np.max(diffs)
        min_diff = np.min(diffs)
        avg_diff = np.mean(diffs)
        median_diff = np.median(diffs)

        condition_flags = {
            "max0": max_diff == 0,
            "max1": max_diff == 1,
            "avg_le1": avg_diff <= 1,
            "median0": median_diff == 0,
            "median_le1": median_diff <= 1
        }

        if condition_flags["max0"]:
            max0_count += 1
            max0_files += len(files)
        if condition_flags["max1"]:
            max1_count += 1
            max1_files += len(files)
        if condition_flags["avg_le1"]:
            avg_le1_count += 1
            avg_le1_files += len(files)
        if condition_flags["median0"]:
            median0_count += 1
            median0_files += len(files)
        if condition_flags["median_le1"]:
            median_le1_count += 1
            median_le1_files += len(files)

        stats = {
            "cluster_id": cluster_id,
            "files": files,
            "max_diff": max_diff,
            "min_diff": min_diff,
            "avg_diff": avg_diff,
            "median_diff": median_diff,
        }
        cluster_stats.append(stats)

    # USEFUL -> print out the clustered tables

    # ranked_clusters = sorted(
    #     [c for c in cluster_stats if c["max_diff"] is not None],
    #     key=lambda x: (-x["max_diff"], -x["median_diff"], -x["avg_diff"], -x["min_diff"])
    # )

    # total_multi_clusters = len(ranked_clusters)
    # total_multi_files = sum(len(c["files"]) for c in ranked_clusters)

    # print(f"\n--- Ranked Clusters with Multiple Files ---")
    # for idx, c in enumerate(ranked_clusters, 1):
    #     print(f"\nRank {idx} - Cluster {c['cluster_id']} (Total {len(c['files'])} tables)")
    #     print(f"Label Difference Stats: max={c['max_diff']}, min={c['min_diff']}, avg={c['avg_diff']:.2f}, median={c['median_diff']}")
    #
    #     # Correct filename extraction for substring analysis
    #     cluster_filenames = [re.sub(r'\.parquet$', '', f.split("_licensed_")[1]) for f in c["files"]]
    #
    #     common_subs = common_substrings(cluster_filenames, min_count=3, min_fraction=0.5, min_length=3)
    #
    #     if common_subs:
    #         print(f"Common substrings (≥4 files or ≥50% files, length ≥3): {common_subs}")
    #     else:
    #         print("No common substrings found meeting threshold.")
    #
    #     for f in c["files"]:
    #         print('- ' + f)
    #
    # single_clusters = [c for c in cluster_stats if c["max_diff"] is None]
    # if single_clusters:
    #     print(f"\n--- Clusters with Single File ---")
    #     for c in single_clusters:
    #         print(f"\nCluster {c['cluster_id']} (1 table)")
    #         for f in c["files"]:
    #             print(f)
    #
    # print(f"\n--- Label Difference Summary (Multi-file Clusters Only) ---")
    # print(f"Clusters with max_diff == 0    : {max0_count} / {total_multi_clusters} ({(max0_count/total_multi_clusters*100):.2f}%) | Total tables: {max0_files} / {total_multi_files} ({(max0_files/total_multi_files*100):.2f}%)")
    # print(f"Clusters with max_diff == 1    : {max1_count} / {total_multi_clusters} ({(max1_count/total_multi_clusters*100):.2f}%) | Total tables: {max1_files} / {total_multi_files} ({(max1_files/total_multi_files*100):.2f}%)")
    # print(f"Clusters with avg_diff <= 1    : {avg_le1_count} / {total_multi_clusters} ({(avg_le1_count/total_multi_clusters*100):.2f}%) | Total tables: {avg_le1_files} / {total_multi_files} ({(avg_le1_files/total_multi_files*100):.2f}%)")
    # print(f"Clusters with median_diff == 0 : {median0_count} / {total_multi_clusters} ({(median0_count/total_multi_clusters*100):.2f}%) | Total tables: {median0_files} / {total_multi_files} ({(median0_files/total_multi_files*100):.2f}%)")
    # print(f"Clusters with median_diff <= 1 : {median_le1_count} / {total_multi_clusters} ({(median_le1_count/total_multi_files*100):.2f}%) | Total tables: {median_le1_files} / {total_multi_files} ({(median_le1_files/total_multi_files*100):.2f}%)")

    return [c["files"] for c in cluster_stats]

def construct_SPO(dataset_path):

    valid_types_path = get_path(dataset_path, "VALID_TYPES_FILE_PATH", dataset='gittab')

    with open(valid_types_path, "rb") as f:
        data = pickle.load(f)

    file_to_labels = data['tables']
    # print(f"Total files loaded: {len(file_to_labels)}")

    filenames = list(file_to_labels.keys())
    # print(len(filenames))

    clusters = []
    current_list = None

    cluster_file_path = get_path(dataset_path, "CLUSTER_FILE_PATH", dataset='gittab')

    with open(cluster_file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("Rank"):
                if current_list:
                    if len(current_list) > 3:
                        clusters.append(current_list)
                current_list = []

            elif line.startswith("- ") and current_list is not None:
                filename = line.lstrip("- ").strip()
                current_list.append(filename)

        if current_list:
            if len(current_list) > 3:
                clusters.append(current_list)

    clusters = sorted(clusters, key=len, reverse=True)

    # Print cluster-wise split and label difference
    # print(f"\nCluster-wise breakdown and label differences:\n")
    #
    # # Counters for cluster conditions
    # max0_count = max1_count = avg_le1_count = median0_count = median_le1_count = 0
    # max0_files = max1_files = avg_le1_files = median0_files = median_le1_files = 0
    # total_multi_clusters = total_multi_files = 0
    #
    # # for cluster_id, files in clusters.items():
    # for cluster_id, files in enumerate(clusters):
    #     print(f"--- Cluster {cluster_id} (Total {len(files)} tables) ---")
    #
    #     for f in files:
    #         print(f)
    #
    #     if len(files) <= 1:
    #         continue
    #
    #     total_multi_clusters += 1
    #     total_multi_files += len(files)
    #
    #     cleaned_label_map = {}
    #     for fname in files:
    #         labels = data['tables'][fname]
    #         cleaned_labels = [re.sub(r'[\d_\-]+', ' ', lbl[2:].lower()).strip() for lbl in labels]
    #         cleaned_label_map[fname] = set(cleaned_labels)
    #
    #     diffs = []
    #     for f1, f2 in combinations(files, 2):
    #         diff_count = len(cleaned_label_map[f1].symmetric_difference(cleaned_label_map[f2]))
    #         diffs.append(diff_count)
    #
    #     max_diff = np.max(diffs)
    #     median_diff = np.median(diffs)
    #     avg_diff = np.mean(diffs)
    #
    #     print(f"Label Difference Stats - max_diff: {max_diff}, median_diff: {median_diff}, avg_diff: {avg_diff:.2f}")
    #
    #     if max_diff == 0:
    #         max0_count += 1
    #         max0_files += len(files)
    #     if max_diff == 1:
    #         max1_count += 1
    #         max1_files += len(files)
    #     if avg_diff <= 1:
    #         avg_le1_count += 1
    #         avg_le1_files += len(files)
    #     if median_diff == 0:
    #         median0_count += 1
    #         median0_files += len(files)
    #     if median_diff <= 1:
    #         median_le1_count += 1
    #         median_le1_files += len(files)
    # Cluster summary
    # print(f"\n--- Label Difference Summary (Multi-file Clusters Only) ---")
    # print(f"Clusters with max_diff == 0    : {max0_count} / {total_multi_clusters} ({(max0_count/total_multi_clusters*100):.2f}%) | Total tables: {max0_files} / {total_multi_files} ({(max0_files/total_multi_files*100):.2f}%)")
    # print(f"Clusters with max_diff == 1    : {max1_count} / {total_multi_clusters} ({(max1_count/total_multi_clusters*100):.2f}%) | Total tables: {max1_files} / {total_multi_files} ({(max1_files/total_multi_files*100):.2f}%)")
    # print(f"Clusters with avg_diff <= 1    : {avg_le1_count} / {total_multi_clusters} ({(avg_le1_count/total_multi_clusters*100):.2f}%) | Total tables: {avg_le1_files} / {total_multi_files} ({(avg_le1_files/total_multi_files*100):.2f}%)")
    # print(f"Clusters with median_diff == 0 : {median0_count} / {total_multi_clusters} ({(median0_count/total_multi_clusters*100):.2f}%) | Total tables: {median0_files} / {total_multi_files} ({(median0_files/total_multi_files*100):.2f}%)")
    # print(f"Clusters with median_diff <= 1 : {median_le1_count} / {total_multi_clusters} ({(median_le1_count/total_multi_clusters*100):.2f}%) | Total tables: {median_le1_files} / {total_multi_files} ({(median_le1_files/total_multi_files*100):.2f}%)")

    SPO = []
    rel_id = 0
    table_to_cluster = dict()
    cluster_o_to_relation = dict()
    for cluster_id, files in enumerate(clusters):
        S = f"?Cluster_{cluster_id+1}"
        seen = set()
        for fname in files:
            for O in data['tables'][fname]:
                O = O[2:]
                if O not in seen:
                    P = f"Relation_{rel_id+1}"
                    SPO.append([S,P,O])
                    seen.add(O)
                    rel_id += 1
                    cluster_o_to_relation[S+'_'+O] = P

        for f in files:
            table_to_cluster[f] = S

    table_to_cluster_path = get_path(dataset_path, "TABLE_TO_CLUSTER_PKL_FILE_PATH", dataset='gittab')
    cluster_o_to_relation_path = get_path(dataset_path, "CLUSTER_TO_RELATION_PKL_FILE_PATH", dataset='gittab')
    with open(table_to_cluster_path, 'wb') as f:
        pickle.dump(table_to_cluster, f)
    with open(cluster_o_to_relation_path, 'wb') as f:
        pickle.dump(cluster_o_to_relation, f)

    return SPO

def load_gittables_metadata(parquet_path):
    resolved_path = os.path.expandvars(parquet_path)
    schema = pq.read_schema(resolved_path)

    if b"gittables" not in schema.metadata:
        raise KeyError(f"'gittables' metadata not found in: {resolved_path}")

    gittables_metadata_bytes = schema.metadata[b"gittables"]
    return json.loads(gittables_metadata_bytes.decode("utf-8"))

def table_gt_info(dataset_path):

    # Step 1: get the clusters based on tf-idf (label co-occurrence)
    valid_types_path = get_path(dataset_path, "VALID_TYPES_FILE_PATH", dataset='gittab')
    raw_tables_path = get_path(dataset_path, "RAW_TABLES_PATH", dataset='gittab')
    with open(valid_types_path, "rb") as f:
        valid_types = pickle.load(f)
    parquet_files = [f for f in os.listdir(raw_tables_path) if f.endswith(".parquet") and f in valid_types['tables']]
    file_to_labels = {filename: valid_types["tables"][filename]  for filename in parquet_files}
    cluster_table_combined_with_ranking(file_to_labels, distance_threshold=1.5)

    # Step 2: filter out clusters with less than 4 tables (merge the cluster if neccessary)
    SPO = construct_SPO(dataset_path)

    return SPO, {}

def main():
    dataset_path = "/apollo/users/dya/dataset/gittable_numeric"
    table_gt_info(dataset_path)

if __name__ == "__main__":
    main()