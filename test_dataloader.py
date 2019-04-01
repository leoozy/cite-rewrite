import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
import h5py
import time
from random import shuffle

class TorchDataLoader(Dataset):
    """Class minibatches from data on disk in HDF5 format"""

    def __init__(self):
        split = 'train'
        self.datafn = os.path.join('../', '%s_imfeats.h5' % split)
        with h5py.File(self.datafn, 'r', swmr=True) as dataset:
            self.phrases = list(dataset['phrases'])
            self.pairs = list(dataset['pairs'])
    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, index):
        with h5py.File(self.datafn, 'r', swmr=True) as dataset:
            im_id = self.pairs[0][index]
            features = np.array(dataset[im_id], np.float32)[:500]

        return features

if __name__ == "__main__":
  #  train_loader = TorchDataLoader()
    #trainLoader = torch.utils.data.DataLoader(train_loader, batch_size=1, shuffle=False, num_workers=8)
    a = []
    train_loader = TorchDataLoader()
    aa = [i for i in range(100)]
    shuffle(aa)
    trainLoader = torch.utils.data.DataLoader(train_loader, batch_sampler= [aa, aa], num_workers=8)
    start = time.time()
    i = 0
    for i , f in enumerate(trainLoader):
        a.append(f)
        if i > 200:
            break
    print(len(a[0]))
    end = time.time()
    print(end - start)
