import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset

class TorchDataLoader(Dataset):
    """Class minibatches from data on disk in HDF5 format"""

    def __init__(self, image_path):
        self.image_path = image_path
        self.images = []
        for _,_,files in os.walk(self.image_path):
            for file in files:
                self.images.append(file)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imagename = self.images[index]
        get_image =  cv2.imread(os.path.join(self.image_path, imagename))
        return get_image
if __name__ == "__main__":

    image_dir = '/media/zhangjl/Seagate Expansion Drive/flickr30k/Flickr30kEntities/Images'
    train_loader = TorchDataLoader(image_dir)
    trainLoader = torch.utils.data.DataLoader(train_loader, batch_size=50, shuffle=True, num_workers=8)
    a = []
    for i, image in enumerate(train_loader):
        a.append(image)
        c =1