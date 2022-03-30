#!/usr/bin/python
#coding:utf-8
import torch.nn as nn
import torch
from torch.nn.functional import interpolate
import torch.nn.functional as F
from self_atten import Self_Atten,ChannelSELayer

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]
        # *在这里进行拆包的操作
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        # shape缩小一半
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        # shape减1
        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        # s.size()为[bs, c, h, w]取第2个元素开始往后的元素，就是[h, w]
        # 将4维的结果view为2维的
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # 这里加了padding，Conv2d尺寸不变，这五个参数一个都不能丢
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, 1, 1)
        # self.maxpool1 = nn.MaxPool2d(2)
        self.resd1_1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.Conv2d(32, 32, 1, 1)
        )
        self.resd1_2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 32, 1, 1)
        )

        self.conv2_1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        # self.maxpool2 = nn.MaxPool2d(2)
        self.resd2_1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.Conv2d(64, 64, 1, 1)
        )
        self.resd2_2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 64, 1, 1)
        )

        self.conv3_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, 1)
        # self.maxpool3 = nn.MaxPool2d(2)
        self.resd3_1 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.Conv2d(128, 128, 1, 1)
        )
        self.resd3_2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(128, 128, 1, 1)
        )

        self.conv4_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, 1, 1)
        # self.maxpool4 = nn.MaxPool2d(2)
        self.resd4_1 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.Conv2d(256, 256, 1, 1)
        )
        self.resd4_2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(256, 256, 1, 1)
        )
        #空间增强注意力
        self.atten = Self_Atten(512, 'relu')
        #全局增强注意力
        self.gatten = ChannelSELayer(512)
        self.conv5_1 = nn.Conv2d(256, 512, 3, 1, 1)
        #self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)

        # self.conv5_1 = nn.Conv2d(256, 512, 3, 1, 1)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)

        # self.upconv6 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv6_1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv6_2 = nn.Conv2d(256, 256, 3, 1, 1)

        # self.upconv7 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv7 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv7_1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(128, 128, 3, 1, 1)

        # self.upconv8 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv8 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv8_1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(64, 64, 3, 1, 1)

        # self.upconv9 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv9 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv9_1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(32, 32, 3, 1, 1)

        self.conv10 = nn.Conv2d(32, 3, 1, 1)
        #self.m = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1_1 = self.resd1_1(conv1)
        pool1_2 = self.resd1_2(conv1)
        pool1 = pool1_1 + pool1_2

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2_1 = self.resd2_1(conv2)
        pool2_2 = self.resd2_2(conv2)
        pool2 = pool2_1 + pool2_2

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu( self.conv3_2(conv3))
        pool3_1 = self.resd3_1(conv3)
        pool3_2 = self.resd3_2(conv3)
        pool3 = pool3_1 + pool3_2

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4_1 = self.resd4_1(conv4)
        pool4_2 = self.resd4_2(conv4)
        pool4 = pool4_1 + pool4_2

        conv5 = self.conv5_1(pool4)
        conv5_0 = self.atten(conv5)
        conv5_1=self.gatten(conv5)
        conv5=conv5_0+conv5_1
        conv5 = self.lrelu(conv5)

        # 共4次上采样+cat
        #upcv6 = self.m(conv5)
        upcv6 = interpolate(conv5, scale_factor=2, mode="bilinear")
        upcv6 = self.conv6(upcv6)
        upcv6 = torch.cat([upcv6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(upcv6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        #upcv7 = self.m(conv6)
        upcv7 = interpolate(conv6, scale_factor=2, mode="bilinear")
        upcv7 = self.conv7(upcv7)
        upcv7 = torch.cat([upcv7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(upcv7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        #upcv8 = self.m(conv7)
        upcv8 = interpolate(conv7, scale_factor=2, mode="bilinear")
        upcv8 = self.conv8(upcv8)
        upcv8 = torch.cat([upcv8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(upcv8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        #upcv9 = self.m(conv8)
        upcv9 = interpolate(conv8, scale_factor=2, mode="bilinear")
        upcv9 = self.conv9(upcv9)
        upcv9 = torch.cat([upcv9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(upcv9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        conv10 = self.conv10(conv9)
        return torch.tanh(conv10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        return torch.max(x * 0.2, x)

if __name__ == '__main__':
    net=Unet()
    a=torch.randn((2,3,480,640))
    b=net(a)
    print(b.shape)

