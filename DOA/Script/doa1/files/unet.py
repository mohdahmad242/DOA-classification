import torch
import torch.nn as nn
from torch.nn import init
import logging
from torchvision import models

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(3,2),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(3,2), padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def double_conv1(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(3,2),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(3,2),stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def double_conv2(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(3,3),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(3,3), padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def double_conv3(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3,padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def double_conv4(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(2, 3),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(2, 3), padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def up_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
    return conv

def up_conv1(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=(3,2),stride=2))
    return conv

def crop_ig(one, two):
    ts = two.size()[3]
    tens = one.size()[3]
    print(one.size()[3], two.size()[3])
    delta = abs(tens - ts)
    delta = delta
    return one[:, :, delta: tens-delta, delta: tens-delta]

class UNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=3):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=(2,1), stride=2)
        self.down_conv_1 = double_conv1(img_ch, 64)
        self.down_conv_2 = double_conv2(64, 128)
        self.down_conv_3 = double_conv3(128, 256)

        self.up_trans_3 = up_conv1(256, 128)
        self.up_conv_3 = double_conv2(256, 128)
        
        self.up_trans_4 = up_conv(128, 64)
        self.up_conv_4 = double_conv4(128, 64)
        
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=output_ch,
            kernel_size=1,stride=1,padding=0)
    
    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        
        # decoder
        x = self.up_trans_3(x5)
        x = self.up_conv_3(torch.cat([x, x3], 1))
        x = self.up_trans_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))
        # output
        x = self.out(x)
    
        return x




# model = models.vgg16()

if __name__ == "__main__":
    image = torch.rand(1, 3, 48, 100)
    model = UNet()
    #print(model)
    print(model)


