import torch
import torch.nn as nn
from torch.nn import init
import logging
from torchvision import models
import numpy as np
import torch.nn.functional as F

def double_conv1(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(10,40),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(7,40),stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def double_conv2(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(9,40),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(7,20), padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv


def up_conv1(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=(1, 9), stride=3))
    return conv
    
def up_conv2(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=(1,9), stride=2))
    return conv

def double_upconv1(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(2,1),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(3,1), padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def double_upconv2(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(4, 1),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(1, 1), padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

class encoder(nn.Module):
    def __init__(self,img_ch=3,output_ch=3):
        super(encoder, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=(2,1), stride=2)
        self.down_conv_1 = double_conv1(img_ch, 64)
        self.down_conv_2 = double_conv2(64, 128)
        self.down_conv_3 = double_conv2(128, 256)
    
    def forward(self, image):
        # encoder
       # int("Encoder =================")
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
        return x5

class decoder(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(decoder, self).__init__()
        self.up_trans_1 = up_conv1(256, 128)
        self.up_conv_1 = double_upconv1(128, 64)
        
        self.up_trans_2 = up_conv2(64, 32)
        self.up_conv_2 = double_upconv2(32, 16)
        
        self.out = nn.Conv2d(
            in_channels=16,
            out_channels=output_ch,
            kernel_size=1,stride=1,padding=0)
    
    def forward(self, image):
        # decoder
        # print("Decoder =================")
        x = self.up_trans_1(image)
        # print("up_trans_1x18, S3, P0  => ", x.size())
        x = self.up_conv_1(x)
        # print("up_conv_3x3, S1, P1    => ", x.size())

        x = self.up_trans_2(x)
        # print("up_trans_2x2, S2, P0   => ", x.size())
        x = self.up_conv_2(x)
        # print("up_conv_2x3, s1, p1    => ", x.size())

        # output
        x = self.out(x)
        # a = nn.Softmax(0)
        # print("Final                  => ", x.size())
        return F.softmax(x, dim=0)



# model = models.vgg16()

from collections import OrderedDict

if __name__ == "__main__":
    image = torch.rand(1, 2, 80, 500)
    en = encoder()
    de = decoder()
    de(en(image))


