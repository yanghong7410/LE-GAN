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
from models import Generator
from models import Unet
from datasets import ImageDataset
from utils import saveimage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=list, default=['../../../datasets/car/'], help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--height', type=int, default=512, help='height of the input image')
parser.add_argument('--width', type=int, default=512, help='width of the input image')
# parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--cuda', default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
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

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
# netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
# netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

low = Tensor(opt.batchSize, opt.input_nc, opt.height, opt.width)
gt = Tensor(opt.batchSize, opt.input_nc, opt.height, opt.width)
#high = Tensor(opt.batchSize, opt.output_nc, opt.height, opt.width)
# Dataset loader
transforms_ = [ transforms.Resize([opt.height, opt.width], Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# # Create output dirs if they don't exist
# if not os.path.exists('output/A'):
#     os.makedirs('output/A')
# if not os.path.exists('output/B'):
#     os.makedirs('output/B')
if not os.path.exists('result/test'):
    os.makedirs('result/test')
s_psnr = 0
s_ssim = 0
for i, batch in enumerate(dataloader):
    # Set model input
    low.copy_(batch['A'])
    gt.copy_(batch['B'])

    # real_A = Variable(input_A.copy_(batch['A']))
    # real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    #fake_high = 0.5*(netG_A2B(low).data + 1.0)
    fake_high = netG_A2B(low)
    fake_high = fake_high.permute(0, 2, 3, 1).data.cpu().numpy()
    high = gt.permute(0, 2, 3, 1).data.cpu().numpy()

    fake_high = np.minimum(np.maximum(fake_high, 0.0), 1.0)
    high = np.minimum(np.maximum(high, 0.0), 1.0)

    save_temp = np.concatenate((fake_high[0, :, :, :], high[0, :, :, :]), axis=1)
    save_temp = save_temp[::-1,:,:]
    temp = Image.fromarray(np.uint8(save_temp * 255))
    temp.save("./result" + "/test/%04d_test.png" % (i))

    temp_out = np.uint8(fake_high[0] * 255)
    temp_high = np.uint8(high[0] * 255)
    psnr = skimage.measure.compare_psnr(np.array(temp_out), np.array(temp_high))
    ssim = skimage.measure.compare_ssim(np.array(temp_out), np.array(temp_high), multichannel=True)
    s_psnr += psnr
    s_ssim += ssim
    print("%d/%d, psnr=%.3f, ssim=%.3f" % (i, len(dataloader), psnr, ssim))
    # sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
print("Test, psnr=%.3f, ssim=%.3f" % (s_psnr / len(dataloader), s_ssim / len(dataloader)))
sys.stdout.write('\n')
###################################
