
import numpy as np

def get_bin_accuracy(gold, predicted):
    correct = 0
    all = 0
    for gold_w ,predict_w in zip(gold, predicted):
        matches = np.sum(gold_w == predict_w)
        correct +=matches
        all += len(gold_w)
    return correct/all


def get_bin_precision(gold, predicted):
    TP = 0
    FP = 0
    for gold_w ,predict_w in zip(gold, predicted):
        TP += np.sum((predict_w == 1) & (gold_w == 1))
        FP += np.sum((predict_w == 1) & (gold_w == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

def get_bin_recall(gold, predicted):
    TP = 0
    FN = 0
    for gold_w ,predict_w in zip(gold, predicted):
        TP += np.sum((predict_w == 1) & (gold_w == 1))
        FN += np.sum((predict_w == 0) & (gold_w == 1))
    precision = TP / (TP + FN) if (TP + FN) > 0 else 0
    return precision

def segments_to_binary(word_segments):

    boundaries = []
    idx = 0
    for segment in word_segments[:-1]:  
        idx += len(segment)
        boundaries.append(idx)
    length = sum(len(seg) for seg in word_segments)
    bin_vec = [1 if i in boundaries else 0 for i in range(length)]
    return np.array(bin_vec)



