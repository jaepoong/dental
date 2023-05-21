import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    
    def __init__(self,in_channel,out_channel,use_bias=False,act=nn.ReLU(),inplace=True):
        super().__init__()
        self.use_bias=use_bias
        self.act=act
        self.inplace=inplace
        self.in_channel=in_channel
        self.out_channel=out_channel
        main=[]
        main.append(nn.Conv2d(self.in_channel,self.out_channel,kernel_size=3,stride=1,padding=1,bias=self.use_bias))
        main.append(nn.BatchNorm2d(self.out_channel,affine=True))
        main.append(nn.ReLU(inplace=self.inplace))
        main.append(nn.Conv2d(self.out_channel,self.out_channel,kernel_size=3,stride=1,padding=1,bias=self.use_bias))
        main.append(nn.BatchNorm2d(self.out_channel,affine=True))
        main.append(nn.ReLU(inplace=self.in_channel))
        self.main = nn.Sequential(*main)

    def forward(self,input):
        return self.main(input)

class ResBlk(nn.Module):
    
    def __init__(self, dim_in, dim_out,
                 normalize=True, downsample=False,use_bias=False,act=nn.LeakyReLU(0.2,inplace=True)):
        super().__init__()
        self.use_bias=use_bias
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.act=act

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1,bias=self.use_bias)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1,bias=self.use_bias)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)
        else:
            self.norm1 = nn.BatchNorm2d(dim_in, affine=True)
            self.norm2 = nn.BatchNorm2d(dim_out, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=self.use_bias)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        x = self.conv1(x)
        if self.normalize:
            x = self.norm1(x)
        x = self.act(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.act(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)
    
    

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
    

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
    

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x