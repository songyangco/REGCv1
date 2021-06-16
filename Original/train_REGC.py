import numpy as np
import REGC_model
import data_loader
import scores
from sklearn import metrics

if __name__ == '__main__':
    DATASET = 'cora'
    features, adj, label_true_te = data_loader.load_fast(DATASET)
    [_, cluster_number] = label_true_te.shape
    label_temp, loss_list = REGC_model.use_REGC(features, adj, cluster_number)
    label_pre = np.array(label_temp.detach())
    label_true = np.argmax(label_true_te, axis=1)

    nmi = metrics.normalized_mutual_info_score(label_true, label_pre)
    acc = scores.acc(label_true, label_pre)
    f1 = scores.macro_f1(label_true, label_pre)
    print('On dataset ' + DATASET + ': acc = ' + str(acc) + '; nmi = ' + str(nmi) + '; macro_f1 = ' + str(f1))
