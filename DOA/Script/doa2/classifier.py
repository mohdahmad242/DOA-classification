import torch
import torch.nn as nn
from torch.nn import init
import logging
from torchvision import models
import numpy as np
import torch.nn.functional as F

def double_conv1(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(2,2),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Conv2d(out_c, out_c, kernel_size=(2,2),stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def double_conv2(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(3,3),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Conv2d(out_c, out_c, kernel_size=(3,3), padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def double_conv3(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(3,3),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Conv2d(out_c, out_c, kernel_size=(3,3), padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class encoder(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(encoder, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=(2,1), stride=2)
        # self.down_conv_1 = double_conv1(img_ch, 64)
        # self.down_conv_2 = double_conv2(64, 128)
        # self.down_conv_3 = double_conv3(128, 256)

        self.RCNN1 = RRCNN_block(img_ch, 64, t=t)
        self.RCNN2 = RRCNN_block(64, 128, t=t)
        self.RCNN3 = RRCNN_block(128, 256, t=t)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(25, 50)
        self.fc2 = nn.Linear(50, 120)
        self.fc3 = nn.Linear(120, 181)
        self.out = nn.Conv2d(
            in_channels=256,
            out_channels=output_ch,
            kernel_size=1,stride=1,padding=0)
    
    def forward(self, image):
        # encoder
       #int("Encoder =================")
        # x1 = self.down_conv_1(image)
        # x1 = self.dropout(x1)
        # # print("Conv3x2, S1, P1        => ", x1.size())
        # x2 = self.max_pool_2x2(x1)
        # # print("max_pool_2x1           => ", x2.size())
        # x3 = self.down_conv_2(x2)
        # x3 = self.dropout(x3)
        # # print("Conv3x3, S1, P1        => ", x3.size())
        # x4 = self.max_pool_2x2(x3)
        # # print("max_pool_2x1           => ", x4.size())
        # x5 = self.down_conv_3(x4)
        # x5 = self.dropout(x5)
        # print("Conv3x3, S1, P1        => ", x5.size())
        x1 = self.RCNN1(image)
        # print("Conv3x2, S1, P1        => ", x1.size())
        x2 = self.max_pool_2x2(x1)
        # print("max_pool_2x1           => ", x2.size())
        x3 = self.RCNN2(x2)
        x3 = self.dropout(x3)
        # print("Conv3x3, S1, P1        => ", x3.size())
        x4 = self.max_pool_2x2(x3)
        # print("max_pool_2x1           => ", x4.size())
        x5 = self.RCNN3(x4)
        x5 = self.dropout(x5)

        x6 = self.fc1(x5)
        x6 = self.dropout(x6)
        x7 = self.fc2(x6)
        x7 = self.dropout(x7)
        x8 = self.fc3(x7)
        x8 = self.dropout(x8)
        x = self.out(x8)
        # print("Final                  => ", x.size())
        return x


if __name__ == "__main__":
    image = torch.rand(1, 3, 8, 100)
    en = encoder()
    en(image)


