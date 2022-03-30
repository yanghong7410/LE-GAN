#!/usr/bin/python
#coding:utf-8
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from models import Discriminator
from models import Unet,Unet_WO
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from utils import saveimage
from datasets1 import ImageDataset,ImageDataset_eval
import os
import skimage
import pytorch_ssim
from torchvision.models.vgg import vgg16

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=3001, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
parser.add_argument('--dataroot', type=list, default=['../low_high_datasets/PNLI_2/train/'], help='root directory of the dataset')
parser.add_argument('--dataeval', type=list, default=['../low_high_datasets/PNLI_2/eval1/'], help='root directory of the dataset')
#parser.add_argument('--dataroot', type=list, default=['../datasets/retinex/our485/', '../datasets/oursTotal/'], help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
#parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--cuda', default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

result_dir = "./result_PNLI_WO/"
checkpoint_dir = "./output_PNLI_WO/"
result_test_dir = "./result_PNLI_WO/test"

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(result_test_dir):
    os.makedirs(result_test_dir)
def ssim_loss(input, gt):
    sm_loss = pytorch_ssim.ssim(input, gt)
    return 1 - sm_loss

vgg = vgg16(pretrained=True)
loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
loss_network = loss_network.cuda()
for param in loss_network.parameters():
    param.requires_grad = False
mse_loss = nn.MSELoss()
print(vgg)

def vgg_loss(input, gt):
    perception_loss = mse_loss(loss_network(input), loss_network(gt))
    return perception_loss

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Unet_WO()
netG_B2A = Unet_WO()

netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

    netG_A2B = nn.DataParallel(netG_A2B)
    netG_B2A = nn.DataParallel(netG_B2A)
    netD_A = nn.DataParallel(netD_A)
    netD_B = nn.DataParallel(netD_B)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
if os.path.exists(checkpoint_dir + 'netG_A2B.pth'):
    pth_A2B = torch.load(checkpoint_dir + 'netG_A2B.pth')
    pth_B2A = torch.load(checkpoint_dir + 'netG_B2A.pth')
    pth_A = torch.load(checkpoint_dir + 'netD_A.pth')
    pth_B = torch.load(checkpoint_dir + 'netD_B.pth')
    print("load model...")
    netG_A2B.load_state_dict(pth_A2B)
    netG_B2A.load_state_dict(pth_B2A)
    netD_A.load_state_dict(pth_A)
    netD_B.load_state_dict(pth_B)
    print("load model successful!")
# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0), (1)),]

transforms_eval = [
                transforms.Resize((480,640)),
                transforms.ToTensor(),
                transforms.Normalize((0), (1)),]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu,drop_last=True)
dataloaderval = DataLoader(ImageDataset_eval(opt.dataeval, transforms_=transforms_eval, unaligned=True), batch_size=1, shuffle=False, num_workers=opt.n_cpu,drop_last=True)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################
psnrbest=0
ssimbest=0
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    netG_A2B.train()
    netD_B.train()
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # print(batch['A'])
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)

        loss_GAN_A2B = criterion_GAN(pred_fake.squeeze(), target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake.squeeze(), target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0
        # loss_ssim_ABA = ssim_loss(recovered_A, real_A)
        # loss_vgg_ABA = vgg_loss(recovered_A, real_A)

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0
        # loss_ssim_BAB = ssim_loss(recovered_B, real_B)
        # loss_vgg_BAB = vgg_loss(recovered_B, real_B) * 0.1
        # Total loss
        # loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_ssim_ABA + loss_ssim_BAB + loss_vgg_ABA + loss_vgg_BAB
        # loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_ssim_ABA + loss_ssim_BAB
        # loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB #+ loss_vgg_ABA + loss_vgg_BAB
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB  #+ loss_identity_A + loss_identity_B
        loss_G.backward()

        optimizer_G.step()
        # 保存图片
        if epoch % 50 == 0:
            rA = real_A.permute(0, 2, 3, 1).data.cpu().numpy()
            fB = fake_B.permute(0, 2, 3, 1).data.cpu().numpy()
            rcA = recovered_A.permute(0, 2, 3, 1).data.cpu().numpy()
            saveA = np.concatenate((rA[0, :, :, :], fB[0, :, :, :], rcA[0, :, :, :]), axis=1)
            rB = real_B.permute(0, 2, 3, 1).data.cpu().numpy()
            fA = fake_A.permute(0, 2, 3, 1).data.cpu().numpy()
            rcB = recovered_B.permute(0, 2, 3, 1).data.cpu().numpy()
            saveB = np.concatenate((rB[0, :, :, :], fA[0, :, :, :], rcB[0, :, :, :]), axis=1)
            save_tmp = np.concatenate((saveA, saveB), axis=0)
            saveimage(save_tmp, result_dir, epoch, i)
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real.squeeze(), target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real.squeeze(), target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        print("epoch: %d, batch: %d, loss_G: %.3f, loss_D: %.3f" % (epoch, i, loss_G, loss_D_A + loss_D_B))
        ###################################
        ###################################

        # Progress report (http://localhost:8097)
        # logger.log({'loss_G': loss_G,
        #             'loss_G_identity': (loss_identity_A + loss_identity_B),
        #             'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
        #             'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B),
        #             #'loss_ssim': (loss_ssim_ABA + loss_ssim_BAB),
        #             #'loss_vgg': (loss_vgg_ABA + loss_vgg_BAB)
        #             },
        #             images={'real_A': real_A[0].item(), 'real_B': real_B[0].item(), 'fake_A': fake_A[0].item(), 'fake_B': fake_B[0].item()})
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_B.step()
    if epoch%100==0:
        netG_A2B.eval()
        s_psnr = 0
        s_ssim = 0
        low = Tensor(opt.batchSize, opt.input_nc, 480, 640)
        gt = Tensor(opt.batchSize, opt.input_nc, 480,640)
        with torch.no_grad():
            for i, batch in enumerate(dataloaderval):
                # Set model input
                low.copy_(batch['A'])
                gt.copy_(batch['B'])

                fake_high = netG_A2B(low)
                fake_high = fake_high.permute(0, 2, 3, 1).data.cpu().numpy()
                high = gt.permute(0, 2, 3, 1).data.cpu().numpy()

                fake_high = np.minimum(np.maximum(fake_high, 0.0), 1.0)
                high = np.minimum(np.maximum(high, 0.0), 1.0)

                save_temp = np.concatenate((fake_high[0, :, :, :], high[0, :, :, :]), axis=1)
                temp = Image.fromarray(np.uint8(save_temp * 255))
                temp.save(result_test_dir+"%04d_test_source.png" % (i))

                temp_out = np.uint8(fake_high[0] * 255)
                temp_high = np.uint8(high[0] * 255)
                psnr = skimage.measure.compare_psnr(np.array(temp_out), np.array(temp_high))
                ssim = skimage.measure.compare_ssim(np.array(temp_out), np.array(temp_high), multichannel=True)
                s_psnr += psnr
                s_ssim += ssim
                print("%d/%d, psnr=%.3f, ssim=%.3f" % (i, len(dataloaderval), psnr, ssim))
                # sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
        print("Test, psnr=%.3f, ssim=%.3f" % (s_psnr / len(dataloaderval), s_ssim / len(dataloaderval)))
        if (s_psnr / len(dataloaderval))>psnrbest:
            torch.save(netG_A2B.state_dict(), checkpoint_dir+'netG_A2B_best.pth')
            torch.save(netD_B.state_dict(), checkpoint_dir+'netD_B_best.pth')
            torch.save(netG_B2A.state_dict(), checkpoint_dir + 'netG_B2A_best.pth')
            torch.save(netD_A.state_dict(), checkpoint_dir + 'netD_A_best.pth')
            psnrbest=(s_psnr / len(dataloaderval))

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), checkpoint_dir+'netG_A2B.pth')
    torch.save(netD_B.state_dict(), checkpoint_dir+'netD_B.pth')
    torch.save(netG_B2A.state_dict(), checkpoint_dir+'netG_B2A.pth')
    torch.save(netD_A.state_dict(), checkpoint_dir+'netD_A.pth')

###################################
