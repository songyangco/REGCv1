import scipy.io as sio
import os


def load_fast(name):
    module_path = os.path.dirname(__file__)
    path = module_path + '/' + name + '.mat'
    data = sio.loadmat(path)  # we save data into mat using sio.savemat(path,{'feature':x, 'adj':a, 'label':np.mat(y)})
    features = data['feature']
    Amatrix = data['adj']
    labels = data['label']
    return features, Amatrix, labels


if __name__ == '__main__':
    load_fast('cora')
