#!/usr/bin/python
#coding:utf-8
import torch
import torch.nn as nn

class Self_Atten(nn.Module):
    def __init__(self, in_dim, activation):
        super(Self_Atten, self).__init__()
        self.activation = activation

        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))   # gamma初始为0，因为最开始只需要依赖局部信息
        self.softmax = nn.Softmax(dim=-1)   # 对最后一维进行Softmax

    def forward(self, x):
        """
        :param x: input feature map(B * C * H * W)
        :return:
            out: self attention value + input feature
            # attention: B * N * N(N = H * W)
        """
        b, c, h, w = x.size()
        f_x = self.f(x).view(b, -1, h*w).permute(0, 2, 1) # b * hw * c//8
        g_x = self.g(x).view(b, -1, h*w) # b * c//8 * hw
        energy = torch.bmm(f_x, g_x) # batch中的矩阵乘法，b * hw * hw
        attention = self.softmax(energy) # b * hw * hw
        h_x = self.h(x).view(b, -1, h*w) # b * c * hw

        out = torch.bmm(h_x, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        out = self.gamma * out + x
        return out#, attention


class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        batch_size, c, h, w = input.size()
        squeeze_tensor = input.view(batch_size, c, -1).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input, fc_out_2.view(a, b, 1, 1))
        return output_tensor