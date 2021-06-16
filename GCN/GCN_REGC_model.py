import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from torch_geometric.nn import GCNConv
import gc
import GCN_data_loader
from sklearn import metrics


class REGC(nn.Module):
    '''
    REGC:
    '''

    def __init__(self, input_dim, out_put_dim):
        super(REGC, self).__init__()
        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm2 = nn.BatchNorm1d(256)

        self.conv1 = GCNConv(input_dim, 256)
        self.conv2 = GCNConv(256, out_put_dim)

    def forward(self, f_in, datac):
        x, y = f_in, datac
        x = self.norm1(x)
        h1 = F.leaky_relu(self.conv1(x, y))
        h1 = self.norm2(h1)
        h2 = F.leaky_relu(self.conv2(h1, y))

        result = F.softmax(h2, dim=1)
        return result


# Normalization of A
def normalization(A, symmetric=True):
    [n, _] = A.shape
    A = A + np.eye(n)  # A = A+I
    d = A.sum(1)  # degree of nodes
    if symmetric:
        D = np.diag(np.power(d, -0.5))  # D = D^-1/2
        return D.dot(A).dot(D)
    else:
        D = np.diag(torch.power(d, -1))  # D=D^-1
        return D.dot(A)


# k-neighbors
def k_neighboors(XXT, k):
    if (k < 0):
        return XXT
    XN = XXT.copy()
    XN.sort(0)
    [n, _] = XXT.shape
    k = np.min([n - 3, k])
    for i in range(n):
        for j in range(n):
            if (XXT[i, j] < XN[-(k + 1), j]):
                XXT[i, j] = 0
    return XXT


def k_neighboors_a(XXT, adj, k):
    with torch.no_grad():
        if (k < 0):
            return XXT
        XN = XXT.copy()
        XN.sort(0)
        [n, _] = XXT.shape
        k = np.min([n - 3, k])
        for i in range(n):
            for j in range(n):
                if (XXT[i, j] < XN[-(k + 1), j]) and adj[i, j] == 0:
                    XXT[i, j] = 0
    return XXT


# Construct W for loss function
def constru_AW(A, X, neigh, feat_map='union'):
    with torch.no_grad():
        if (feat_map == 'k-nearest'):
            XXT = X.dot(X.T)
            XXT = k_neighboors(XXT, neigh)
            XXT = 0.5 * (XXT + XXT.T)
            XXT = XXT * (1.0 / np.max(XXT))
        elif (feat_map == 'union'):
            XXT = X.dot(X.T)
            XXT = k_neighboors_a(XXT, A, neigh)
            XXT = 0.5 * (XXT + XXT.T)
            XXT = XXT * (1.0 / np.max(XXT))
        elif (feat_map == 'addition'):
            XXT = X.dot(X.T)
            A_USE = A.copy()
            A_USE[A_USE != 0] = -1
            A_USE[A_USE != -1] = 1
            A_USE[A_USE != 1] = 0
            XXC = np.multiply(XXT, A_USE)
            A_S = k_neighboors(XXC, neigh)
            A_all = A_S + A
            A_all[A_all != 0] = 1
            XXT = np.multiply(XXT, A_all)
            XXT = XXT * (1.0 / np.max(XXT))
        else:  # initial
            XXT = A

        [n, _] = XXT.shape
        for i in range(n):
            XXT[i, i] = 0.0
        return XXT


# residual entropy loss
class ETR_loss_trace(nn.Module):
    def __init__(self):
        super(ETR_loss_trace, self).__init__()

    def forward(self, A, C, IsumC, IsumCDC):  # A:input weight; C: indicator; Isum: matrix made up by ones
        total = torch.trace((1 / (torch.sum(A))) * (C.t().mm(A.mm(C))).mul(
            torch.log2(IsumCDC.mm(IsumC.mm((A.mm(C))) * (1 / (torch.sum(A)))))))

        return total


def use_REGC(datac, features_in, adj_in, c_num, feat_map='union', neigh=25):
    '''
    :param features_in: input attributes
    :param adj_in:  input adjacency (initial)
    :param c_num:  community number
    :param feat_map: neighbor strategy
    :param neigh:  neighbor number
    :return:
    '''
    # learning rate, epochs
    learning_rate = 0.01
    epochs = 1000

    # find gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    datac = datac.to(device)

    _, X_dim = features_in.shape
    model = REGC(X_dim, c_num).to(device)
    # use residual entropy loss
    criterion = ETR_loss_trace().to(device)
    # Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    lr_optim = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8, verbose=False,
                                                          threshold=0.000001, threshold_mode='rel', cooldown=0,
                                                          min_lr=1e-4, eps=1e-08)
    # preprocessing
    x = features_in
    x = x.astype(np.float32)
    tensor_x = torch.from_numpy(x).to(device)
    normalize_adj = normalization(adj_in)
    normalize_adj = normalize_adj.astype(np.float32)
    tensor_adj = torch.from_numpy(normalize_adj).to(device)

    loss_adj = constru_AW(normalize_adj, features_in, neigh, feat_map)
    loss_adj = loss_adj.astype(np.float32)
    tensor_loss_adj = torch.from_numpy(loss_adj).to(device)

    # out put loss list
    loss_history = []
    # trainning mode
    model.train()
    for epoch in range(epochs):
        gc.collect()
        result = model(tensor_x, datac)
        # 为loss提供tensor
        n, k = list(result.size())
        IsumC = torch.ones(1, n).to(device)
        IsumCDC = torch.ones(k, 1).to(device)
        loss = criterion(tensor_loss_adj, result, IsumC, IsumCDC)
        _, y_out = result.max(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_optim.step(loss)
        loss_history.append(loss.item())
    # fixed mode
    model.eval()
    with torch.no_grad():
        result = model(tensor_x, datac)
        _, y_out = result.max(1)
    if torch.cuda.is_available():
        y_out = y_out.cpu()
    return y_out, loss_history  # output indicator and loss list
