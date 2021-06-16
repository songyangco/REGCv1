import scipy.io as sio
import os
from torch_geometric.datasets import Planetoid
from collections import defaultdict
import numpy as np
import torch


def load_fast(name):
    module_path = os.path.dirname(__file__)
    path = module_path + '/' + name + '.mat'
    data = sio.loadmat(path)  # we save data into mat using sio.savemat(path,{'feature':x, 'adj':a, 'label':np.mat(y)})
    features = data['feature']
    Amatrix = data['adj']
    labels = data['label']
    nodeone = []
    nodetwo = []
    for i in range(2708):
        for j in range(len(Amatrix[i])):
            if Amatrix[i][j] != 0:
                nodeone += [int(i)]
                nodetwo += [int(j)]
    edge_index = torch.FloatTensor([nodeone, nodetwo])
    edge_index = edge_index.long()
    return features, Amatrix, labels, edge_index
