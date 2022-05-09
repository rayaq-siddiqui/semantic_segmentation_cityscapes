import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

class CityScapesDataset(Dataset):

    def __init__(self, img_dir, km):
        """
        :param path: path to image files
        :param km: kmeans object for dimensioality reduction
        """
        super().__init__()
        self.img_dir = img_dir
        self.paths= os.listdir(img_dir)
        self.km = km


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img, seg = self.load_image(name=str(img_path), path=self.img_dir)
        seg = self.colors_to_class(seg, self.km)
        return img, seg


    def load_image(self, name, path, rotation=0.0, flip=False, 
                   cut_bottom=58,size=(256, 200)):
        """
        Loads image
        """
        img = Image.open(str(path)+"/"+str(name))
        img = np.array(img)
        seg = img[:-cut_bottom, 256:]
        img = img[:-cut_bottom, 0:256]

        for i in range(3):
            zimg = img[:,:,i]
            zimg = cv2.equalizeHist(zimg)
            img[:,:,i] = zimg

        img = Image.fromarray(img).resize(size)
        seg = Image.fromarray(seg).resize(size)
        img = img.rotate(rotation)
        seg = seg.rotate(rotation)

        img = np.array(img)
        seg = np.array(seg)

        if flip:
            img = img[:,::-1,:]
            seg = seg[:,::-1,:]

        #seg = np.round(seg/255.0)

        return img/255, seg/255


    def colors_to_class(self, seg, km):
        """
        Maps color to class based on KMeans Clustering on Trainign Data

        :param seg: the mask
        :param km: kmeans object for dimensioality reduction
        """
        s = seg.reshape((seg.shape[0]*seg.shape[1],3))
        s = km.predict(s)
        s = s.reshape((seg.shape[0], seg.shape[1]))

        n = len(km.cluster_centers_)

        # tocategorical - one hot encoding each pixel
        cls = np.zeros((seg.shape[0], seg.shape[1], n))

        for i in range(n):
            m = np.copy(s)
            m[m!=i] = 0
            m[m!=0] = 1
            cls[:,:,i]=m
        
        return cls
