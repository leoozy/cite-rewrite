import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset

class TorchDataLoader(Dataset):
    """Class minibatches from data on disk in HDF5 format"""

    def __init__(self, image_path, test_dict):
        self.image_path = image_path
        self.images = []
        self.dict = test_dict
     #   for _,_,files in os.walk(self.image_path):
     #       for file in files:
      #          self.images.append(file)
    def __len__(self):
        return 4

    def __getitem__(self, index):
        #imagename = self.images[index]
        get_image =  cv2.imread(os.path.join(self.image_path, imagename))

        return get_image
def changedict(dict):
    train_loader = TorchDataLoader(image_dir, dict)
    dict = {
        1: 10,
        2: 20,
        3: 30,
        4: 40
    }
    trainLoader = torch.utils.data.DataLoader(train_loader, batch_size=1, shuffle=False, num_workers=8)
    a = []
    for i, image in enumerate(train_loader):
        a.append(image)
        c =1
if __name__ == "__main__":

    image_dir = '/media/zhangjl/Seagate Expansion Drive/flickr30k/Flickr30kEntities/Images'
    dict = {}
    changedict(dict)
