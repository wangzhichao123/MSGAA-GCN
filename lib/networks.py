import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from gcn import Grapher
from gltb import GLTB
from fusion import Fusion
from maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
from maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
from maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out


logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def load_pretrained_weights(img_size, model_scale):
    
    if(model_scale=='tiny'):
        if img_size==224:
            backbone = maxvit_tiny_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './pretrained_pth/maxvit/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
        elif(img_size==256):
            backbone = maxvit_rmlp_tiny_rw_256_4out()
            print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
    elif(model_scale=='small'):
        if img_size==224:
            backbone = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', 'maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
            state_dict = torch.load('maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        elif(img_size==256):
            backbone = maxxvit_rmlp_small_rw_256_4out()
            print('Loading:', 'maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
            state_dict = torch.load('maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
    else:
        sys.exit(model_scale+" is not a valid model scale! Currently supported model scales are 'tiny' and 'small'.")
        
    backbone.load_state_dict(state_dict, strict=False)
    print('Pretrain weights loaded.')
    
    return backbone

class MERIT_GLAG_Cascaded(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), model_scale='small', interpolation='bilinear'):
        super(MERIT_GLAG_Cascaded, self).__init__()
        
        self.n_class = n_class
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.model_scale = model_scale
        self.interpolation = interpolation
        
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone1 = load_pretrained_weights(self.img_size_s1[0], self.model_scale)
        self.backbone2 = load_pretrained_weights(self.img_size_s2[0], self.model_scale)
        
        if(self.model_scale=='tiny'):
            self.channels = [512, 256, 128, 64]
        elif(self.model_scale=='small'):
            self.channels = [768, 384, 192, 96]

        # ------------- GCN Decoder1 ---------------

        self.gcn4_1 = Grapher(768, 9, 1, 'mr', 'gelu', 'batch', False, False, 0.2, 1, n=64, drop_path=0.0, relative_pos=True)
        self.gcn3_1 = Grapher(384, 9, 1, 'mr', 'gelu', 'batch', False, False, 0.2, 2, n=256, drop_path=0.0, relative_pos=True)
        self.gcn2_1 = Grapher(192, 9, 4, 'mr', 'gelu', 'batch', False, False, 0.2, 4, n=1024, drop_path=0.0, relative_pos=True)
        self.gcn1_1 = Grapher(96, 9, 2, 'mr', 'gelu', 'batch', False, False, 0.2, 8, n=4096, drop_path=0.0, relative_pos=True)
        
        self.gltb4_1 = GLTB(768)
        self.gltb3_1 = GLTB(384)
        self.gltb2_1 = GLTB(192)
        self.gltb1_1 = GLTB(96)
        
        self.fusion3_1 = Fusion(384, 768)
        self.fusion2_1 = Fusion(192, 384)
        self.fusion1_1 = Fusion(96, 192)
        
        # ------------- GCN Decoder2 ---------------

        self.gcn4_2 = Grapher(768, 9, 1, 'mr', 'gelu', 'batch', False, False, 0.2, 1, n=49, drop_path=0.0, relative_pos=True)
        self.gcn3_2 = Grapher(384, 9, 1, 'mr', 'gelu', 'batch', False, False, 0.2, 2, n=196, drop_path=0.0, relative_pos=True)
        self.gcn2_2 = Grapher(192, 9, 4, 'mr', 'gelu', 'batch', False, False, 0.2, 4, n=784, drop_path=0.0, relative_pos=True)
        self.gcn1_2 = Grapher(96, 9, 2, 'mr', 'gelu', 'batch', False, False, 0.2, 8, n=3136, drop_path=0.0, relative_pos=True)
        
        self.gltb4_2 = GLTB(768)
        self.gltb3_2 = GLTB(384)
        self.gltb2_2 = GLTB(192)
        self.gltb1_2 = GLTB(96)
        
        self.fusion3_2 = Fusion(384, 768)
        self.fusion2_2 = Fusion(192, 384)
        self.fusion1_2 = Fusion(96, 192)
        
            
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)
        
        self.out_head1_1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2_1 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3_1 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4_1 = nn.Conv2d(self.channels[3], self.n_class, 1)

        self.out_head4_in = nn.Conv2d(self.channels[3], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # transformer backbone as encoder
        if(x.shape[2]%14!=0):
            f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
        else:
            f1 = self.backbone2(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
               
        # print([f1[3].shape,f1[2].shape,f1[1].shape,f1[0].shape])
        # [torch.Size([1, 768, 8, 8]), torch.Size([1, 384, 16, 16]), torch.Size([1, 192, 32, 32]), torch.Size([1, 96, 64, 64])]
        
        
        gcn4_1 = self.gcn4_1(f1[3])
        gltb4_1 = self.gltb4_1(gcn4_1)        
        
        skip3_1 = self.fusion3_1(f1[2], gltb4_1) 
        
        
        gcn3_1 = self.gcn3_1(skip3_1)
        gltb3_1 = self.gltb3_1(gcn3_1) 
        
        skip2_1 = self.fusion2_1(f1[1], gltb3_1)
        
        
        gcn2_1 = self.gcn2_1(skip2_1)    
        gltb2_1 = self.gltb2_1(gcn2_1)
            
        skip1_1 = self.fusion1_1(f1[0], gltb2_1)
        
        gcn1_1 = self.gcn1_1(skip1_1)
        gltb1_1 = self.gltb1_1(gcn1_1)
        
        # decoder
        # x11_o, x12_o, x13_o, x14_o = self.decoder(f1[3], [f1[2], f1[1], f1[0]])
        # torch.Size([1, 768, 8, 8]) torch.Size([1, 384, 16, 16]) torch.Size([1, 192, 32, 32]) torch.Size([1, 96, 64, 64])
        # print(x11_o.shape,x12_o.shape,x13_o.shape,x14_o.shape)

#         # prediction heads  
        p11 = self.out_head1(gltb4_1)
        p12 = self.out_head2(gltb3_1)
        p13 = self.out_head3(gltb2_1)
        p14 = self.out_head4(gltb1_1)
        # torch.Size([1, 9, 8, 8]) torch.Size([1, 9, 16, 16]) torch.Size([1, 9, 32, 32]) torch.Size([1, 9, 64, 64])  解码不改变图片大小，只是变为9通道
        # print(p11.shape,p12.shape,p13.shape,p14.shape)
#         # calculate feedback from 1st decoder
        p14_in = self.out_head4_in(gltb1_1)
        p14_in = self.sigmoid(p14_in)
        
        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)
        
#         #print([p11.shape,p12.shape,p13.shape,p14.shape])

        p14_in = F.interpolate(p14_in, scale_factor=4, mode=self.interpolation)
        # print(p14_in.shape) torch.Size([1, 1, 256, 256])
        
#         # apply feedback from 1st decoder to input
        x_in = x * p14_in + x
        # print(x_in.shape) torch.Size([1, 3, 256, 256])
         
        
        if(x.shape[2]%14!=0):
            f2 = self.backbone2(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))
        else:
            f2 = self.backbone1(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))
            
        skip1_0 = F.interpolate(f1[0], size=(f2[0].shape[-2:]), mode=self.interpolation)
        skip1_1 = F.interpolate(f1[1], size=(f2[1].shape[-2:]), mode=self.interpolation)
        skip1_2 = F.interpolate(f1[2], size=(f2[2].shape[-2:]), mode=self.interpolation)
        skip1_3 = F.interpolate(f1[3], size=(f2[3].shape[-2:]), mode=self.interpolation)
        
        gcn4_2 = self.gcn4_2(f2[3]+skip1_3)
        gltb4_2 = self.gltb4_2(gcn4_2)      
        
        skip3_2 = self.fusion3_2(f2[2]+skip1_2, gltb4_2)
        
        
        gcn3_2 = self.gcn3_2(skip3_2)
        gltb3_2 = self.gltb3_2(gcn3_2)   
        
        skip2_2 = self.fusion2_2(f2[1]+skip1_1, gltb3_2)
        
        
        gcn2_2 = self.gcn2_2(skip2_2)    
        gltb2_2 = self.gltb2_2(gcn2_2)
            
        skip1_2 = self.fusion1_2(f2[0]+skip1_0, gltb2_2)
        
        gcn1_2 = self.gcn1_2(skip1_2)
        gltb1_2 = self.gltb1_2(gcn1_2)
        
        # print(skip1_0.shape,skip1_1.shape,skip1_2.shape,skip1_3.shape)
        # torch.Size([1, 96, 56, 56]) torch.Size([1, 192, 28, 28]) torch.Size([1, 384, 14, 14]) torch.Size([1, 768, 7, 7])
        
#         x21_o, x22_o, x23_o, x24_o = self.decoder(f2[3]+skip1_3, [f2[2]+skip1_2, f2[1]+skip1_1, f2[0]+skip1_0])

        p21 = self.out_head1_1(gltb4_2)
        p22 = self.out_head2_1(gltb3_2)
        p23 = self.out_head3_1(gltb2_2)
        p24 = self.out_head4_1(gltb1_2)

        # print([p21.shape,p22.shape,p23.shape,p24.shape])
              
        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)
        
        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
#         #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        return p1, p2, p3, p4

class MERIT_MSGAA_Cascaded(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), model_scale='small', interpolation='bilinear'):
        super(MERIT_MSGAA_Cascaded, self).__init__()
        
        self.n_class = n_class
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.model_scale = model_scale 
        self.decoder_aggregation = decoder_aggregation      
        self.interpolation = interpolation
        
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone1 = load_pretrained_weights(self.img_size_s1[0], self.model_scale)
        self.backbone2 = load_pretrained_weights(self.img_size_s2[0], self.model_scale)
        
        if(self.model_scale=='tiny'):
            self.channels = [512, 256, 128, 64]
        elif(self.model_scale=='small'):
            self.channels = [768, 384, 192, 96]
            
        # ------------- GCN Decoder1 ---------------

        self.gcn4_1 = Grapher(768, 9, 1, 'mr', 'gelu', 'batch', False, False, 0.2, 1, n=64, drop_path=0.0, relative_pos=True)
        self.gcn3_1 = Grapher(384, 9, 1, 'mr', 'gelu', 'batch', False, False, 0.2, 2, n=256, drop_path=0.0, relative_pos=True)
        self.gcn2_1 = Grapher(192, 9, 4, 'mr', 'gelu', 'batch', False, False, 0.2, 4, n=1024, drop_path=0.0, relative_pos=True)
        self.gcn1_1 = Grapher(96, 9, 2, 'mr', 'gelu', 'batch', False, False, 0.2, 8, n=4096, drop_path=0.0, relative_pos=True)
    
        
        # self.gltb4_1 = GLTB(768)
        # self.gltb3_1 = GLTB(384)
        # self.gltb2_1 = GLTB(192)
        # self.gltb1_1 = GLTB(96)
        
        # 256 Input
        self.gltb4_1 = GLTB1(in_c=768, num_heads=8, num_patches=64, qkv_bias=True, sr_ratio=1, agent_num=64)     # 8
        self.gltb3_1 = GLTB1(in_c=384, num_heads=4, num_patches=256, qkv_bias=True, sr_ratio=1, agent_num=64)    # 16
        self.gltb2_1 = GLTB1(in_c=192, num_heads=2, num_patches=1024, qkv_bias=True, sr_ratio=1, agent_num=9)   # 32
        self.gltb1_1 = GLTB1(in_c=96,  num_heads=1, num_patches=4096, qkv_bias=True, sr_ratio=1, agent_num=4)    # 64
        
        self.fusion3_1 = Fusion(384, 768)
        self.fusion2_1 = Fusion(192, 384)
        self.fusion1_1 = Fusion(96, 192)
        
        # ------------- GCN Decoder2 ---------------

        self.gcn4_2 = Grapher(768, 9, 1, 'mr', 'gelu', 'batch', False, False, 0.2, 1, n=49, drop_path=0.0, relative_pos=True)
        self.gcn3_2 = Grapher(384, 9, 1, 'mr', 'gelu', 'batch', False, False, 0.2, 2, n=196, drop_path=0.0, relative_pos=True)
        self.gcn2_2 = Grapher(192, 9, 4, 'mr', 'gelu', 'batch', False, False, 0.2, 4, n=784, drop_path=0.0, relative_pos=True)
        self.gcn1_2 = Grapher(96, 9, 2, 'mr', 'gelu', 'batch', False, False, 0.2, 8, n=3136, drop_path=0.0, relative_pos=True)
        
        # self.gltb4_2 = GLTB(768)
        # self.gltb3_2 = GLTB(384)
        # self.gltb2_2 = GLTB(192)
        # self.gltb1_2 = GLTB(96)
        
        self.gltb4_2 = GLTB1(in_c=768, num_heads=8, num_patches=49, qkv_bias=True, sr_ratio=1, agent_num=49)
        self.gltb3_2 = GLTB1(in_c=384, num_heads=4, num_patches=196, qkv_bias=True, sr_ratio=1, agent_num=49)
        self.gltb2_2 = GLTB1(in_c=192, num_heads=2, num_patches=784, qkv_bias=True, sr_ratio=1, agent_num=16)
        self.gltb1_2 = GLTB1(in_c=96,  num_heads=1, num_patches=3136, qkv_bias=True, sr_ratio=1, agent_num=9)
        
        self.fusion3_2 = Fusion(384, 768)
        self.fusion2_2 = Fusion(192, 384)
        self.fusion1_2 = Fusion(96, 192)
        
        self.d4 = nn.Sequential(nn.Conv2d(768, 768, kernel_size=3, padding=1), nn.BatchNorm2d(768), nn.GELU())
        self.d3 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.GELU())
        self.d2 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.BatchNorm2d(192), nn.GELU())
        self.d1 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.GELU())
            
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

        self.out_head4_in = nn.Conv2d(self.channels[3], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # transformer backbone as encoder
        # print([f1[3].shape,f1[2].shape,f1[1].shape,f1[0].shape])
        # [torch.Size([1, 768, 8, 8]), torch.Size([1, 384, 16, 16]), torch.Size([1, 192, 32, 32]), torch.Size([1, 96, 64, 64])]
        if(x.shape[2]%14!=0):
            f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
        else:
            f1 = self.backbone2(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
        
        
        gcn4_1 = self.gcn4_1(f1[3])
        gltb4_1 = self.gltb4_1(gcn4_1)        
        
        skip3_1 = self.fusion3_1(f1[2], gltb4_1) 
        
        
        gcn3_1 = self.gcn3_1(skip3_1)
        gltb3_1 = self.gltb3_1(gcn3_1) 
        
        skip2_1 = self.fusion2_1(f1[1], gltb3_1)
        
        
        gcn2_1 = self.gcn2_1(skip2_1)    
        gltb2_1 = self.gltb2_1(gcn2_1)
            
        skip1_1 = self.fusion1_1(f1[0], gltb2_1)
        
        gcn1_1 = self.gcn1_1(skip1_1)
        gltb1_1 = self.gltb1_1(gcn1_1)
        
        # decoder
        # x11_o, x12_o, x13_o, x14_o = self.decoder(f1[3], [f1[2], f1[1], f1[0]])
        # torch.Size([1, 768, 8, 8]) torch.Size([1, 384, 16, 16]) torch.Size([1, 192, 32, 32]) torch.Size([1, 96, 64, 64])
        # print(x11_o.shape,x12_o.shape,x13_o.shape,x14_o.shape)

#         # prediction heads  
        p11 = self.out_head1(gltb4_1)
        p12 = self.out_head2(gltb3_1)
        p13 = self.out_head3(gltb2_1)
        p14 = self.out_head4(gltb1_1)
        # torch.Size([1, 9, 8, 8]) torch.Size([1, 9, 16, 16]) torch.Size([1, 9, 32, 32]) torch.Size([1, 9, 64, 64]) 
        # print(p11.shape,p12.shape,p13.shape,p14.shape)
#         # calculate feedback from 1st decoder
        p14_in = self.out_head4_in(gltb1_1)
        p14_in = self.sigmoid(p14_in)
        
        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)
        

        p14_in = F.interpolate(p14_in, scale_factor=4, mode=self.interpolation)
        # print(p14_in.shape) torch.Size([1, 1, 256, 256])
        # apply feedback from 1st decoder to input
        x_in = x * p14_in + x
        # print(x_in.shape) torch.Size([1, 3, 256, 256])
         
        
        if(x.shape[2]%14!=0):
            f2 = self.backbone2(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))
        else:
            f2 = self.backbone1(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))
            
        skip1_0 = F.interpolate(f1[0], size=(f2[0].shape[-2:]), mode=self.interpolation)
        skip1_1 = F.interpolate(f1[1], size=(f2[1].shape[-2:]), mode=self.interpolation)
        skip1_2 = F.interpolate(f1[2], size=(f2[2].shape[-2:]), mode=self.interpolation)
        skip1_3 = F.interpolate(f1[3], size=(f2[3].shape[-2:]), mode=self.interpolation)
        
        
        gcn4_2 = self.gcn4_2(f2[3]+self.d4(abs(f2[3]-skip1_3)))
        gltb4_2 = self.gltb4_2(gcn4_2)      
        
        skip3_2 = self.fusion3_2(f2[2]+skip1_2, gltb4_2)
        
        
        gcn3_2 = self.gcn3_2(skip3_2+self.d3(abs(f2[2]-skip1_2)))
        gltb3_2 = self.gltb3_2(gcn3_2)   
        
        skip2_2 = self.fusion2_2(f2[1]+skip1_1, gltb3_2)
        
        
        gcn2_2 = self.gcn2_2(skip2_2+self.d2(abs(f2[1]-skip1_1)))    
        gltb2_2 = self.gltb2_2(gcn2_2)
            
        skip1_2 = self.fusion1_2(f2[0]+skip1_0, gltb2_2)
        
        gcn1_2 = self.gcn1_2(skip1_2+self.d1(abs(f2[0]-skip1_0)))
        gltb1_2 = self.gltb1_2(gcn1_2)
        
        # print(skip1_0.shape,skip1_1.shape,skip1_2.shape,skip1_3.shape)
        # torch.Size([1, 96, 56, 56]) torch.Size([1, 192, 28, 28]) torch.Size([1, 384, 14, 14]) torch.Size([1, 768, 7, 7])
        
#         x21_o, x22_o, x23_o, x24_o = self.decoder(f2[3]+skip1_3, [f2[2]+skip1_2, f2[1]+skip1_1, f2[0]+skip1_0])

        p21 = self.out_head1(gltb4_2)
        p22 = self.out_head2(gltb3_2)
        p23 = self.out_head3(gltb2_2)
        p24 = self.out_head4(gltb1_2)

        # print([p21.shape,p22.shape,p23.shape,p24.shape])
              
        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)
        
        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
#         #print([p1.shape,p2.shape,p3.shape,p4.shape])
        
        return p1, p2, p3, p4

        
if __name__ == '__main__':
    model = MERIT_GLAG_Cascaded(9, img_size_s1=(256,256), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()

    p1, p2, p3, p4 = model(input_tensor)
    print(p1.shape, p2.shape, p3.shape, p4.shape)

