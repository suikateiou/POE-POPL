import copy
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_var_df(df, var):
    return df[var]
    # return df[var_cols].to_numpy()


def cat(data_list, axis=1):
    try:
        output = torch.cat(data_list, axis)
    except:
        output = np.concatenate(data_list, axis)

    return output


def split(data, split_ratio=0.5):
    data1 = copy.deepcopy(data)
    data2 = copy.deepcopy(data)

    split_num = int(data.length * split_ratio)
    data1.split(0, split_num)
    data2.split(split_num, data.length)

    return data1, data2


class CausalDataset(object):
    def __init__(self, dataset, dataset_path, train=True):
        if train:
            filename = "train.npz"
        else:
            filename = "validation.npz"
        self.data_path = dataset_path / filename

        self.data = getDataset(np.load(self.data_path))

        self.statistics = {
            'jobs': {
                'x': (-4, 29.1),
                't': (0.5, 4),
                's': (1.5, 5.9),
                'y': (0.8, 5.3)
            },
            'ihdp': {
                'x': (-5.2, 3),
                't': (-1.3, 8.2),
                's': (2.3, 8.75),
                'y': (3.4, 10.7)
            },
            'twins': {
                'x': (0, 2),
                't': (0.5, 4.7),
                's': (1.6, 3.4),
                'y': (2.4, 3.6)
            },
            'simulation': {
                'x': (0, 2),
                't': (1.3, 4.3),
                's': (0.6, 1.8),
                'y': (1.3, 1.8)
            },
            'crime': {
                'x': (0, 1),
                't': (0, 1),
                's': (0, 1),
                'y': (0, 1)
            }
        }

        self.X_range = self.statistics[dataset]['x']
        self.T_range = self.statistics[dataset]['t']
        self.S_range = self.statistics[dataset]['s']
        self.Y_range = self.statistics[dataset]['y']

    @staticmethod
    def normalization(data, range_val):
        (min_val, max_val) = range_val
        normalized_data = (data - min_val) / (max_val - min_val)
        normalized_data = 2 * (normalized_data - 0.5)
        return normalized_data

    @staticmethod
    def denormalization(normalized_data, range_val):
        (min_val, max_val) = range_val
        denormalized_data = (normalized_data / 2) + 0.5
        denormalized_data = denormalized_data * (max_val - min_val) + min_val
        return denormalized_data

    @staticmethod
    def normalize_columns(matrix):
        norms = np.linalg.norm(matrix, axis=0)
        normalized_matrix = matrix / norms
        return normalized_matrix

    def get_data(self):
        return self.data

    def tensor(self):
        self.data.tensor()

    def double(self):
        self.data.double()

    def float(self):
        self.data.float()

    def detach(self):
        self.data.detach()

    def to(self, device='cpu'):
        self.data.to(device)

    def cpu(self):
        self.data.cpu()

    def numpy(self):
        self.data.numpy()

    def __len__(self):
        return self.data.length

    def __getitem__(self, index):
        X = torch.from_numpy(self.data[index]['x'].astype('float32'))
        # X = torch.from_numpy(self.trans(self.data[index]['x'].astype('float32')))
        # X = F.normalize(X, p=2, dim=1)
        T = torch.from_numpy(self.normalization(self.data[index]['t'].astype('float32'), self.T_range))
        S = torch.from_numpy(self.data[index]['s'].astype('float32'))
        Y = torch.from_numpy(self.data[index]['y'].astype('float32'))

        return X, T, S, Y

    def get_min_max(self):
        return self.data.s_min, self.data.s_max, self.data.y_min, self.data.y_max


class getDataset(Dataset):
    def __init__(self, array):
        self.length = array['x'].shape[0]
        self.Vars = array.files

        for var in self.Vars:
            exec(f'self.{var}=get_var_df(array, \'{var}\')')

        # if not hasattr(self, 'i'):
        #     self.i = self.z
        #     self.Vars.append('i')

        self.x_dim = array['x'][0].shape[0]
        self.s_min = np.min(array["s"])
        self.s_max = np.max(array["s"])
        self.y_min = np.min(array["y"])
        self.y_max = np.max(array["y"])

    def split(self, start, end):
        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}[start:end]')
            except:
                pass

        self.length = end - start

    def cpu(self):
        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}.cpu()')
            except:
                break

    def cuda(self, n=0):
        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}.cuda({n})')
            except:
                break

    def to(self, device='cpu'):
        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}.to(\'{device}\')')
            except:
                break

    def tensor(self):
        for var in self.Vars:
            try:
                exec(f'self.{var} = torch.Tensor(self.{var})')
            except:
                break

    def float(self):
        for var in self.Vars:
            try:
                exec(f'self.{var} = torch.Tensor(self.{var}).float()')
            except:
                break

    def double(self):
        for var in self.Vars:
            try:
                exec(f'self.{var} = torch.Tensor(self.{var}).double()')
            except:
                break

    def detach(self):
        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}.detach()')
            except:
                break

    def numpy(self):
        try:
            self.detach()
        except:
            pass

        try:
            self.cpu()
        except:
            pass

        for var in self.Vars:
            try:
                exec(f'self.{var} = self.{var}.numpy()')
            except:
                break

    def pandas(self, path=None):
        var_list = []
        var_dims = []
        var_name = []
        for var in self.Vars:
            exec(f'var_list.append(self.{var})')
            exec(f'var_dims.append(self.{var}.shape[1])')
        for i in range(len(self.Vars)):
            for d in range(var_dims[i]):
                var_name.append(self.Vars[i] + str(d))
        df = pd.DataFrame(np.concatenate(var_list, axis=1), columns=var_name)

        if path is not None:
            df.to_csv(path, index=False)
        return df

    def __getitem__(self, idx):
        var_dict = {}
        for var in self.Vars:
            exec(f'var_dict[\'{var}\']=self.{var}[idx]')

        return var_dict

    def __len__(self):
        return self.length
