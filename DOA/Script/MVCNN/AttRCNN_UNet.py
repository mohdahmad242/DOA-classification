import torch
import torch.nn as nn
from torch.nn import init

def single_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
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


def up_conv1(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=(2, 2), stride=2))
    return conv
    
def up_conv2(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=(2,2), stride=2))
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


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi



class Att_R2U(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(Att_R2U, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.RCNN1 = RRCNN_block(img_ch, 64, t=t)
        self.RCNN2 = RRCNN_block(64, 128, t=t)
        self.RCNN3 = RRCNN_block(128, 256, t=t)

        self.up_trans_1 = up_conv1(256, 128)
        self.Att1 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN1 = RRCNN_block(256, 128,t=t)
        
        self.up_trans_2 = up_conv2(128, 64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(128, 64,t=t)
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=output_ch,
            kernel_size=1,stride=1,padding=0)

    def forward(self, image):
        # encoder
        # print("Input Image            => ", image.size())
        # print("Encoder =================")
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
        # print("Conv3x3, S1, P1        => ", x5.size())
        
        
        # decoder
        # print("Decoder =================")
        x = self.up_trans_1(x5)
        # print("up_trans_1x18, S3, P0  => ", x.size())
        x3 = nn.functional.interpolate(x3, (x.size()[2], x.size()[3]))
        x3 = self.Att1(g=x,x=x3)
        x = self.Up_RRCNN1(torch.cat([x, x3], 1))
        x = self.dropout(x)
        # print("up_conv_3x3, S1, P1    => ", x.size())

        x = self.up_trans_2(x)
        # print("up_trans_2x2, S2, P0   => ", x.size())
        x1 = nn.functional.interpolate(x1, (x.size()[2], x.size()[3]))
        x1 = self.Att2(g=x,x=x1)
        x = self.Up_RRCNN2(torch.cat([x, x1], 1))
        x = self.dropout(x)
        # print("up_conv_2x3, s1, p1    => ", x.size())
        # output
        x = self.out(x)
        # print(x.size())
        return x



if __name__ == "__main__":
    print("start")
    image = torch.rand(1, 1, 8, 100)
    model = Att_R2U()
    model(image)