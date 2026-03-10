import os
import glob
import json
import filecmp
import pickle

import matplotlib.pyplot as plt
from collections import Counter

import langid

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def make_plot(data, fname, range):

    plt.hist(data, range=range, bins=250)
    plt.xlabel('Values')
    plt.ylabel('Frequencies')
    plt.title('Histogram')
    plt.savefig(fname)
    plt.figure().clear()
    plt.cla()

def check_cell_length(load_fn, save_fn):
    table_lens = []
    table_dict = load(load_fn)
    for table_id, token_id_list in table_dict.items():
        for cell in token_id_list:
            table_lens.append(len(cell))

    range = (0, 25)
    def cnr(hist, threshold):
        count = 0
        for i in hist:
            if i <= threshold:
                count += 1
        return count

    print("- lens that are no more than 5 : {}".format(cnr(table_lens, 5)))
    print("- lens that are no more than 10 : {}".format(cnr(table_lens, 10)))
    print("- lens that are no more than 15 : {}".format(cnr(table_lens, 15)))
    print("- total : {}".format(len(table_lens)))

    make_plot(table_lens, save_fn, range)

def check_lang(load_fn, save_fn):

    _dict = {}
    table_dict = load(load_fn)

    nt = 0
    for _, row_list in table_dict.items():
        for row in row_list:
            line = " ".join([ v for k, v in row.items()])
            lang = langid.classify(line)[0]
            if lang in _dict:
                _dict[lang] += 1
            else:
                _dict[lang] = 1
            nt += 1
            break

    print ( "Non-English: {}".format(nt - _dict['en']))
    print ( "English: {}".format(_dict['en']))

    plt.pie(list(_dict.values()), labels=list(_dict.keys()), autopct='%1.1f%%')
    plt.savefig(save_fn)
    plt.figure().clear()
    plt.cla()

def check_rows(load_fn, save_fn):

    table_dict = load(load_fn)

    hist = []
    range = (0,250)
    for _, row_list in table_dict.items():
        rows_len = len(row_list)
        hist.append(rows_len)
    make_plot(hist, save_fn, range)

    def cnr(hist, threshold):
        count = 0
        for i in hist:
            if i <= threshold:
                count += 1
        return count

    print("- rows that are no more than 40 : {}".format(cnr(hist, 40)))
    print("- rows that are no more than 80 : {}".format(cnr(hist, 80)))
    print("- rows that are no more than 120 : {}".format(cnr(hist, 120)))
    print("- total : {}".format(len(hist)))


def check_cols(load_fn, save_fn):

    table_dict = load(load_fn)

    hist = []
    range = (0,30)
    for _, row_list in table_dict.items():
        max_col = len(row_list[0])
        hist.append(max_col)
    make_plot(hist, save_fn, range)

    def cnr(hist, threshold):
        count = 0
        for i in hist:
            if i <= threshold:
                count += 1
        return count

    print("- cols that are no more than 5 : {}".format(cnr(hist, 5)))
    print("- cols that are no more than 10 : {}".format(cnr(hist, 10)))
    print("- cols that are no more than 15 : {}".format(cnr(hist, 15)))
    print("- total : {}".format(len(hist)))

def main():
    
    dataset_path = "/apollo/users/dya/dataset/semtab"

    cta_pkl_table_fn = os.path.join(dataset_path, 'cta_table.pkl')
    cpa_pkl_table_fn = os.path.join(dataset_path, 'cpa_table.pkl')
    cta_pkl_table_lm_tokenized_fn = os.path.join(dataset_path, 'cta.pkl')
    cpa_pkl_table_lm_tokenized_fn = os.path.join(dataset_path, 'cpa.pkl')

    cta_col_fn = "CTA_PNG1"
    cta_row_fn = "CTA_PNG2"
    cta_lang_fn = "CTA_PNG3"
    cta_cell_len_fn = "CTA_PNG4"

    cpa_col_fn = "CPA_PNG1"
    cpa_row_fn = "CPA_PNG2"
    cpa_lang_fn = "CPA_PNG3"
    cpa_cell_len_fn = "CPA_PNG4"

    check_cols(cta_pkl_table_lm_tokenized_fn, cta_col_fn)
    check_rows(cta_pkl_table_lm_tokenized_fn, cta_row_fn)
    check_lang(cta_pkl_table_fn, cta_lang_fn)
    check_cell_length(cta_pkl_table_lm_tokenized_fn, cta_cell_len_fn)

    check_cols(cpa_pkl_table_lm_tokenized_fn, cpa_col_fn)
    check_rows(cpa_pkl_table_lm_tokenized_fn, cpa_row_fn)
    check_lang(cpa_pkl_table_fn, cpa_lang_fn)
    check_cell_length(cpa_pkl_table_lm_tokenized_fn, cpa_cell_len_fn)

if __name__ == '__main__':
    main()