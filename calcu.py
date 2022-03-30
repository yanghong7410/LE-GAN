#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import skimage
import numpy as np
from PIL import Image
from models import Unet
from models import Unet
from datasets import ImageDataset_eval
from utils import saveimage

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default=['../low_high_datasets/PNLI_2/eval1/'], help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--height', type=int, default=480, help='height of the input image')
parser.add_argument('--width', type=int, default=640, help='width of the input image')
# parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--cuda', default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output_LOL/netG_A2B_best.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output_SYN/netG_B2A_best.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Unet()
#netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    #netG_B2A.cuda()

import time
from thop import profile
with torch.no_grad():
    netG_A2B.eval()
    input = torch.randn((1, 3, 400,608 )).cuda()
    start = time.time()
    for i in range(100):
        out =netG_A2B(input)

    end_time = (time.time() - start)
    print(end_time / 100)
    flops, params = profile(netG_A2B, inputs=(input,))

    print('the flops is {}M,the params is {}G'.format(round(flops / (10 ** 9), 2),
                                                      round(params / (10 ** 6),
                                                            2)))  # 4111514624.0 25557032.0 res50