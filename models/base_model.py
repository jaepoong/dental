import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from models.block import *
from torch.nn import functional as F_torch

## TODO 이 generator랑 discriminator 정리해서 모듈 수좀 줄이자... 너무 중구난방하게 실험했다....
class BaseGenerator(nn.Module):
    def __init__(self,n_res=4,use_bias=False):
        super().__init__()
        
        self.n_res=n_res
        self.use_bias=use_bias
        self.down_sampling=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=7,stride=1,padding=1,bias=use_bias),
            nn.BatchNorm2d(64,affine=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1,bias=use_bias),
            nn.BatchNorm2d(128,affine=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1,bias=use_bias),
            nn.BatchNorm2d(256,affine=True),
            nn.LeakyReLU(0.2,inplace=True),
        )
        
        res_block=[]
        for i in range(n_res):
            res_block.append(ResBlk(256,256,normalize=True,use_bias=self.use_bias))
        self.res_block=nn.Sequential(*res_block) #75,162
        
        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=0, output_padding=0, bias=use_bias),
            nn.BatchNorm2d(128,affine=True),
            nn.LeakyReLU(0.2,inplace=True),

            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(128,affine=True),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(128, 1, kernel_size=5, stride=1, padding=0, bias=use_bias),
            nn.Tanh()
        )
    
    def forward(self,input):
        x=self.down_sampling(input)
        x=self.res_block(x)
        out=self.up_sampling(x)
        return out
    
class U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64) 
        self.Conv2 = conv_block(ch_in=64,ch_out=128) 
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        #self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        #self.Up5 = up_conv(ch_in=1024,ch_out=512)
        #self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x): #512,512
        # encoding path
        x1 = self.Conv1(x) #512,512,64

        x2 = self.Maxpool(x1) #256,256,64
        x2 = self.Conv2(x2) #256,256,128
        
        x3 = self.Maxpool(x2) #128,128,128
        x3 = self.Conv3(x3) #128,128,256

        x4 = self.Maxpool(x3) #64,64,256
        x4 = self.Conv4(x4) #64,64,512

        #x5 = self.Maxpool(x4)#32,32,512
        #x5 = self.Conv5(x5)#32,32,1024

        # decoding + concat path
        #d5 = self.Up5(x5) #64,64,512
        #d5 = torch.cat((x4,d5),dim=1) #64,64,1024
        
        #d5 = self.Up_conv5(d5) #64,64,512
        
        d4 = self.Up4(x4) #128,128,256
        d4 = torch.cat((x3,d4),dim=1) #128,128,512
        d4 = self.Up_conv4(d4) #128,128,256

        d3 = self.Up3(d4) #256.256.128
        d3 = torch.cat((x2,d3),dim=1) #256,256,256
        d3 = self.Up_conv3(d3) #256.256.128

        d2 = self.Up2(d3) #512.512.64
        d2 = torch.cat((x1,d2),dim=1) #512,512,128
        d2 = self.Up_conv2(d2) #512,512,64

        d1 = self.Conv_1x1(d2) #512,512,3

        return d1

class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = nn.Conv2d(128,256,3,1,1,bias=True )#conv_block(ch_in=128,ch_out=256)
        self.Conv4 = nn.Conv2d(256,512,3,1,1,bias=True)
        #self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        #self.Up5 = up_conv(ch_in=1024,ch_out=512)
        #self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        #self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = nn.Conv2d(512, 256,3,1,1,bias=True)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = nn.Conv2d(256,128,3,1,1,bias=True) #conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        #x5 = self.Maxpool(x4)
        #x5 = self.Conv5(x5)

        # decoding + concat path
        #d5 = self.Up5(x5)
        #x4 = self.Att5(g=d5,x=x4)
        #d5 = torch.cat((x4,d5),dim=1)        
        #d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1 

class BaseDiscriminator(nn.Module):
    def __init__(self,use_bias=False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0, bias=use_bias),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, bias=use_bias),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0, bias=use_bias)

        )

    def forward(self, input):
        output = self.layers(input)
        # adding average pooling
        #output=F_torch.avg_pool2d(output,output.size()[2:])
        return output