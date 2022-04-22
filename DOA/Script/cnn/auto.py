import torch
import torch.nn as nn
from torch.nn import init
import logging
from torchvision import models
import numpy as np
import torch.nn.functional as F

def double_conv1(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(5,4),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(8,3),stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def double_conv2(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(9,3),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(7,3), padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv


class encoder(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(encoder, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=(2,1), stride=2)
        self.down_conv_1 = double_conv1(img_ch, 64)
        self.down_conv_2 = double_conv2(64, 128)
        self.down_conv_3 = double_conv2(128, 256)
        self.out = nn.Conv2d(
            in_channels=256,
            out_channels=output_ch,
            kernel_size=1,stride=1,padding=0)
        self.fc1 = nn.Linear(25, 50)
        self.fc2 = nn.Linear(50, 120)
        self.fc3 = nn.Linear(120, 181)
    def forward(self, image):
        # encoder
       #int("Encoder =================")
        x1 = self.down_conv_1(image)
        # print("Conv3x2, S1, P1        => ", x1.size())
        x2 = self.max_pool_2x2(x1)
        # print("max_pool_2x1           => ", x2.size())
        x3 = self.down_conv_2(x2)
        # print("Conv3x3, S1, P1        => ", x3.size())
        x4 = self.max_pool_2x2(x3)
        # print("max_pool_2x1           => ", x4.size())
        x5 = self.down_conv_3(x4)
        # print("Conv3x3, S1, P1        => ", x5.size())
        # outt = x5.reshape(x5.size(0), -1)
        x6 = self.fc1(x5)
        x7 = self.fc2(x6)
        x8 = self.fc3(x7)
        # print("fc1, S1, P1        => ", x6.size())
        # x7 = self.fc2(x6)
        # print("fc2, S1, P1        => ", x7.size())
        out = self.out(x8)
        # print(out.size())
        return out





if __name__ == "__main__":
    image = torch.rand(1, 3, 80, 100)
    en = encoder()
    en(image)


