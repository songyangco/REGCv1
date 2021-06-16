import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics


def bestmap(y_true, y_pred):
    assert len(y_pred) == len(y_true)
    D = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    ind, tid = linear_assignment(np.max(w) - w)
    y_aim = y_pred
    for i in range(len(y_aim)):
        y_aim[i] = tid[y_aim[i]]
    return y_aim


def macro_f1(y_true, y_pred):
    '''
    Calculate macro_f1. unsupervised models require best map
    :param y_true:
    :param y_pred:
    :return:
    '''

    label_pre = bestmap(y_true, y_pred)
    f1score = metrics.f1_score(y_true, label_pre, average='macro')
    return f1score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. unsupervised models require best map
     y: true labels, numpy.array with shape `(n_samples,)`
     y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    """
    assert len(y_pred) == len(y_true)
    D = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    ind, tid = linear_assignment(np.max(w) - w)

    flag = 0
    for i in range(len(y_pred)):
        if (ind[y_true[i]] == tid[y_pred[i]]):
            flag = flag + 1
    return flag / len(y_pred)
