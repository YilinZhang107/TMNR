import h5py
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y,  clean_Y):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.conf = dict()
        if isinstance(data_X, dict):
            self.views_num = len(data_X)
        else:
            self.views_num = data_X.shape[0]
        for v in range(self.views_num):
            self.X[v] = self.normalize(data_X[v])

        self.Y = data_Y
        self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.clean_Y = clean_Y

        self.classes_num = len(np.unique(self.Y))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = {}
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        clean_Y = self.clean_Y[index]
        return data, target, index, clean_Y

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.views_num):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x


