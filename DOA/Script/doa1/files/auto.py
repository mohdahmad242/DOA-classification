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
        nn.Conv2d(in_c, out_c, kernel_size=(4,3),padding=1 ,stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Conv2d(out_c, out_c, kernel_size=(4,3), padding=1 ,stride=1, bias=True),
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


def up_conv1(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=(1, 19), stride=3))
    return conv
    
def up_conv2(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=(1,1), stride=2))
    return conv

def double_upconv1(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(3,3),padding=1 , bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(3,3), padding=1 , bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def double_upconv2(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(1, 1) , bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Conv2d(out_c, out_c, kernel_size=(1, 1) , bias=True),
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
from AttRCNN_UNet import Att_R2U
preTrained = Att_R2U()


pre_model = torch.load("./SNR_50000_dropout.pth")
preTrained.load_state_dict(pre_model, strict=False)


for prams in preTrained.parameters():
  prams.requires_grad = False

class encoder(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(encoder, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=(3,2), stride=2)
        self.down_conv_1 = double_conv1(img_ch, 64)
        self.down_conv_2 = double_conv2(64, 128)
        self.down_conv_3 = double_conv3(128, 256)
        self.down_conv_4 = double_conv3(256, 512)
        # self.RCNN1 = RRCNN_block(img_ch, 64, t=t)
        # self.RCNN2 = RRCNN_block(64, 128, t=t)
        # self.RCNN3 = RRCNN_block(128, 256, t=t)
        self.unet = preTrained
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(50, 100)
        self.fc2 = nn.Linear(100, 140)
        self.fc3 = nn.Linear(140, 181)
        self.out = nn.Conv2d(
            in_channels=512,
            out_channels=output_ch,
            kernel_size=1,stride=1,padding=0)
    
    def forward(self, image):
        # encoder
        #int("Encoder =================")
        # image = self.unet(image) 
        # trained_channels = []
        # for i in range(0,3):
        #     a = image[:, i, :, :]

        #     a = a.unsqueeze(1)
        #     # print(a.size(), i)
            
            
        #     # print(cnn1_out.size())
        #     trained_channels.append(x)
        # stack_ch = torch.stack(trained_channels)
        # # print(stack_ch.size())
        # pooled_views, _ = torch.max(stack_ch, dim=0)
        # x = pooled_views

        x = self.down_conv_1(image)
        x = self.dropout(x)
        x = self.max_pool_2x2(x)

        x = self.down_conv_2(x)
        x = self.dropout(x)
        x = self.down_conv_3(x)
        x = self.dropout(x)
        # print("RCNN3        => ", x.size())

        x = self.down_conv_4(x)
        x = self.dropout(x)
        # print("RCNN3        => ", x.size())
        # print("max_pool_2x1           => ", x.size())
        # x = self.RCNN4(x)
        # x = self.dropout(x)
        # x = self.RCNN5(x)
        # x = self.dropout(x)
        # x = self.RCNN6(x)
        # x = self.dropout(x)
        
        # print("max_pool_2x1           => ", x.size())
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # x = self.dropout(x)
        # x = self.out(x)
        print("Final                  => ", x.size())
        return x

class decoder(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(decoder, self).__init__()
        self.up_trans_1 = up_conv1(256, 128)
        self.up_conv_1 = double_upconv1(128, 64)
        
        self.up_trans_2 = up_conv2(64, 32)
        self.up_conv_2 = double_upconv2(32, 16)
        self.dropout = nn.Dropout(p=0.3)
        
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
        x = self.dropout(x)
        # print("up_conv_3x3, S1, P1    => ", x.size())

        x = self.up_trans_2(x)
        x = self.dropout(x)
        # print("up_trans_2x2, S2, P0   => ", x.size())
        x = self.up_conv_2(x)
        x = self.dropout(x)
        # print("up_conv_2x3, s1, p1    => ", x.size())

        # output
        x = self.out(x)
        # print(x.size())
        # a = nn.Softmax(0)
        print("Final                  => ", x.size())
        return x



# model = models.vgg16()

from collections import OrderedDict

if __name__ == "__main__":
    image = torch.rand(1, 3, 8, 100)
    en = encoder()
    en(image)


