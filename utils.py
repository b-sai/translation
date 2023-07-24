import numpy as np
import math
from copy import deepcopy
# import torch


def get_score(a, b, freq_table, substition=False):
    if a == b and substition:
        return 0.000000001
    if(a, b) in freq_table:
        return freq_table[(a, b)]
    else:
        return 9e2


def enhanced_levenshtein(s1, s2, ins_table, del_table, sub_table):

    s1 = "#"+s1
    s2 = "#"+s2

    rows = []

    previous_row = [0]
    for i in range(1, len(s2)):
        previous_row.append(
            previous_row[i-1] + get_score(s2[i-1], s2[1], freq_table=ins_table))
    rows.append(previous_row)
    for i in range(1, len(s1)):
        current_row = [
            get_score(s1[i], s1[i-1], freq_table=del_table)+previous_row[0]]
        for j in range(1, len(s2)):
            ins_score = get_score(
                s2[j], s1[i], freq_table=ins_table) + current_row[j-1]
            del_score = get_score(
                s1[i], s1[i-1], freq_table=del_table)+previous_row[j]
            sub_score = get_score(
                s1[i], s2[j], freq_table=sub_table, substition=True)+previous_row[j-1]
            current_row.append(min(ins_score, del_score, sub_score))
        rows.append(current_row)
        previous_row = current_row
    return previous_row[-1]/max(len(s1), len(s2))


def get_enhanced_preds(y_mod, pred_vec: np.array, src_word: str,  ins_table, del_table, sub_table):
    pred_words = y_mod.most_similar(pred_vec, topn=400)
    words = [item[0] for item in pred_words]
    avg_scores = []
    for word, cos_dist in pred_words:
        score = enhanced_levenshtein(
            src_word, word, ins_table, del_table, sub_table)
        avg_scores.append((1-cos_dist)+(score/40))
    # avg_idxs = torch.topk(torch.tensor(avg_scores), 10,
    #                       sorted=True, largest=False)[1]
    avg_idxs = np.sort(np.argpartition(avg_scores, 10)[:10])
    avg_preds = [words[i] for i in avg_idxs]
    return avg_preds
