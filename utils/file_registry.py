"""
Centralized filename registry.

Every script in graph/, tokenizers/, and loaders/ looks up filenames here

"""

import os

# ──────────────────────────────────────────────
# SOTAB
# ──────────────────────────────────────────────
SOTAB_FILES = {
    # Label vocabularies
    "CTA_DBP_LABEL_TXT":     "raw_data/CTA-DBP-Datasets/cta_labels_round2_dbpedia.txt",
    "CPA_DBP_LABEL_TXT":     "raw_data/CPA-DBP-Datasets/cpa_labels_round2_dbpedia.txt",
    "CTA_SCH_LABEL_TXT":     "raw_data/CTA-SCH-Datasets_Merged/cta_labels.txt",
    "CPA_SCH_LABEL_TXT":     "raw_data/CPA-SCH-Datasets_Merged/cpa_labels.txt",

    # Ground truth — DBpedia
    "CTA_DBP_TRAIN_GT_CSV":  "raw_data/CTA-DBP-Datasets/sotab_cta_train_round2_dbpedia.csv",
    "CTA_DBP_VAL_GT_CSV":    "raw_data/CTA-DBP-Datasets/sotab_cta_validation_round2_dbpedia.csv",
    "CTA_DBP_TEST_GT_CSV":   "raw_data/CTA-DBP-Datasets/sotab_cta_test_dbpedia.csv",
    "CPA_DBP_TRAIN_GT_CSV":  "raw_data/CPA-DBP-Datasets/sotab_cpa_train_round2_dbpedia.csv",
    "CPA_DBP_VAL_GT_CSV":    "raw_data/CPA-DBP-Datasets/sotab_cpa_validation_round2_dbpedia.csv",
    "CPA_DBP_TEST_GT_CSV":   "raw_data/CPA-DBP-Datasets/sotab_cpa_test_dbpedia.csv",

    # Ground truth — Schema.org
    "CTA_SCH_TRAIN_GT_CSV":  "raw_data/CTA-SCH-Datasets_Merged/sotab_v2_cta_training_set.csv",
    "CTA_SCH_VAL_GT_CSV":    "raw_data/CTA-SCH-Datasets_Merged/sotab_v2_cta_validation_set.csv",
    "CTA_SCH_TEST_GT_CSV":   "raw_data/CTA-SCH-Datasets_Merged/sotab_v2_cta_test_set.csv",
    "CPA_SCH_TRAIN_GT_CSV":  "raw_data/CPA-SCH-Datasets_Merged/sotab_v2_cpa_training_set.csv",
    "CPA_SCH_VAL_GT_CSV":    "raw_data/CPA-SCH-Datasets_Merged/sotab_v2_cpa_validation_set.csv",
    "CPA_SCH_TEST_GT_CSV":   "raw_data/CPA-SCH-Datasets_Merged/sotab_v2_cpa_test_set.csv",
}

# ──────────────────────────────────────────────
# TURL (WikiTables)
# ──────────────────────────────────────────────
TURL_FILES = {
    "CTA_TURL_LABEL_TXT":    "type_vocab.txt",
    "CPA_TURL_LABEL_TXT":    "relation_vocab.txt",

    "TURL_TRAIN_TABLE_JSON":  "train_tables.jsonl",
    "TURL_DEV_TABLE_JSON":    "dev_tables.jsonl",

    "CTA_TURL_TRAIN_GT_JSON": "train.table_col_type.json",
    "CTA_TURL_DEV_GT_JSON":   "dev.table_col_type.json",
    "CPA_TURL_TRAIN_GT_JSON": "train.table_rel_extraction.json",
    "CPA_TURL_DEV_GT_JSON":   "dev.table_rel_extraction.json",

    # WikiTables-specific intermediate files
    "CTA_PG":                 "graph/cta_pg.pkl",
    "CPA_PG":                 "graph/cpa_pg.pkl",
    "SUBJ_DICT":              "graph/subj_dict.pkl",
    "OBJ_DICT":               "graph/obj_dict.pkl",
    "TRIPLE_PKL":             "graph/kg_0101_3.pkl",
    "P_SPO_DICT_PKL":         "graph/p_spo_dict_4.pkl",
}

# ──────────────────────────────────────────────
# GitTab
# ──────────────────────────────────────────────
GITTAB_FILES = {
    "RAW_TABLES_PATH":                     "raw_data",
    "CTA_GIT_LABEL_TXT":                   "cta_labels.txt",
    "SYNTHETIC_REL_LABEL_TXT":             "synthetic_rel_labels.txt",

    "CTA_GIT_TRAIN_GT_CSV":                "train_cta_annotations.csv",
    "CTA_GIT_VAL_GT_CSV":                  "dev_cta_annotations.csv",
    "CTA_GIT_TEST_GT_CSV":                 "test_cta_annotations.csv",
    "CPA_GIT_TRAIN_GT_CSV":                "train_rel_annotations.csv",
    "CPA_GIT_VAL_GT_CSV":                  "dev_rel_annotations.csv",
    "CPA_GIT_TEST_GT_CSV":                 "test_rel_annotations.csv",

    # GitTab-specific intermediate files
    "CLUSTER_FILE_PATH":                   "gold_graph.txt",
    "TABLE_TO_CLUSTER_PKL_FILE_PATH":      "table_to_cluster.pkl",
    "CLUSTER_TO_RELATION_PKL_FILE_PATH":   "cluster_o_to_relation.pkl",
    "VALID_TYPES_FILE_PATH":               "valid_types_tables.pkl",
}


def get_path(dataset_path, key, dataset="sotab"):
    """
    Look up a filename by its placeholder key and return the full path.
    """
    registry = {
        "sotab":  SOTAB_FILES,
        "turl":   TURL_FILES,
        "gittab": GITTAB_FILES,
    }

    files = registry.get(dataset)
    if files is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(registry.keys())}")

    filename = files.get(key)
    if filename is None:
        raise KeyError(f"Unknown file key '{key}' for dataset '{dataset}'. Known keys: {list(files.keys())}")

    return os.path.join(dataset_path, filename)