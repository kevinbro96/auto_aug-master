import numpy as np
import io
import json
import os
import pickle
import PIL.Image
import scipy.stats
import torchvision
from torch.utils.data import Dataset, DataLoader
import pathlib
import torch
def load_new_test_data(version_string='', load_tinyimage_indices=False):
    data_path = os.path.join(os.path.dirname(__file__), './datasets/')
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    print('Loading labels from file {}'.format(label_filepath))
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    print('Loading image data from file {}'.format(imagedata_filepath))
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6' or version_string == 'v7':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021

    if not load_tinyimage_indices:
        return imagedata, labels
    else:
        ti_indices_data_path = os.path.join(os.path.dirname(__file__), '../other_data/')
        ti_indices_filename = 'cifar10.1_' + version_string + '_ti_indices.json'
        ti_indices_filepath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
        print('Loading Tiny Image indices from file {}'.format(ti_indices_filepath))
        assert pathlib.Path(ti_indices_filepath).is_file()
        with open(ti_indices_filepath, 'r') as f:
            tinyimage_indices = json.load(f)
        assert type(tinyimage_indices) is list
        assert len(tinyimage_indices) == labels.shape[0]
        return imagedata, labels, tinyimage_indices

class cifar101(Dataset):
    def __init__(self, version, transform):
        self.data, self.label = load_new_test_data(version)
        self.transform = transform

    def __getitem__(self,i):
        img = self.data[i,:,:,:]
        img = self.transform(img)
        label = self.label[i]
        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        datalen = len(self.label)
        return datalen


