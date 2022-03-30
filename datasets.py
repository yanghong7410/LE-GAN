#!/usr/bin/python
#coding:utf-8
import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        # self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        # self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        #print(root)
        #print(os.path.join(root, 'low'))
        self.files_A = []
        self.files_B = []

        for i in range(len(root)):
            self.files_A.extend(sorted(glob.glob(os.path.join(root[i], 'low') + '/*.*')))
            self.files_B.extend(sorted(glob.glob(os.path.join(root[i], 'high') + '/*.*')))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        # tmpA = cv2.imread(self.files_A[index % len(self.files_A)])
        # item_A = self.transform(tmpA[:,:,::-1])
        # if self.unaligned:
            # tmpB = cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)])
            # item_B = self.transform(tmpB[:,:,::-1])
        item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        # else:
        #     # tmpB = cv2.imread(self.files_B[index % len(self.files_B)])
        #     # item_B = self.transform(tmpB[:, :, ::-1])
        #     item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset_eval(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        # self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        # self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        #print(root)
        #print(os.path.join(root, 'low'))
        self.files_A = []
        self.files_B = []

        for i in range(len(root)):
            self.files_A.extend(sorted(glob.glob(os.path.join(root[i], 'low') + '/*.*')))
            self.files_B.extend(sorted(glob.glob(os.path.join(root[i], 'high') + '/*.*')))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index]).convert('RGB'))

        # tmpA = cv2.imread(self.files_A[index % len(self.files_A)])
        # item_A = self.transform(tmpA[:,:,::-1])
        # if self.unaligned:
            # tmpB = cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)])
            # item_B = self.transform(tmpB[:,:,::-1])
        item_B = self.transform(Image.open(self.files_B[index]).convert('RGB'))
        # else:
        #     # tmpB = cv2.imread(self.files_B[index % len(self.files_B)])
        #     # item_B = self.transform(tmpB[:, :, ::-1])
        #     item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B,'Aname':self.files_A[index]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))