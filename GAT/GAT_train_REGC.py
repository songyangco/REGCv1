import numpy as np
import GAT_REGC_model
import GAT_data_loader
import scores
from sklearn import metrics
import statistics
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nmi_list = []
    acc_list = []
    f1_list = []
    DATASET = 'cora'
    for n in range(20):
        print("epoch: ", n)
        features, adj, label_true_te, data = GAT_data_loader.load_fast(DATASET)
        [_, cluster_number] = label_true_te.shape
        label_temp, loss_list = GAT_REGC_model.use_REGC(data, features, adj, cluster_number)
        label_pre = np.array(label_temp.detach())
        label_true = np.argmax(label_true_te, axis=1)

        nmi = metrics.normalized_mutual_info_score(label_true, label_pre)
        acc = scores.acc(label_true, label_pre)
        f1 = scores.macro_f1(label_true, label_pre)
        nmi_list += [nmi]
        acc_list += [acc]
        f1_list += [f1]

    print("nmi", np.average(nmi_list), statistics.pvariance(nmi_list), statistics.pstdev(nmi_list))
    print("acc", np.average(acc_list), statistics.pvariance(acc_list), statistics.pstdev(acc_list))
    print("f1", np.average(f1_list), statistics.pvariance(f1_list), statistics.pstdev(f1_list))
