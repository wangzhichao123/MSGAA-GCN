import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leakyrelu2(x)
        return x
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x1_2 = torch.cat((x1, x2), dim=1)
        out = self.conv(x1_2)
        return out
class DeepUNet7(nn.Module):
    def __init__(self, n_class=9):
        super(DeepUNet7, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(64, 64)
        self.conv_21 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lr1 = nn.LeakyReLU(negative_slope=0.01)
        self.conv4 = ConvBlock(160, 128)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(128, 128)
        self.conv7 = ConvBlock(64, 64)

        self.conv8 = ConvBlock(128, 128)
        self.conv9 = ConvBlock(128, 128)
        self.conv10 = ConvBlock(64, 64)
        self.conv_41 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lr2 = nn.LeakyReLU(negative_slope=0.01)
        self.conv11 = ConvBlock(288, 256)

        self.conv12 = ConvBlock(256, 512)
        self.conv13 = ConvBlock(256, 256)
        self.conv14 = ConvBlock(128, 128)
        self.conv15 = ConvBlock(128, 128)
        self.conv16 = ConvBlock(64, 64)

        self.bn3 = nn.BatchNorm2d(32)
        self.lr3 = nn.LeakyReLU(negative_slope=0.01)

        self.conv17 = ConvBlock(256, 256)
        self.conv18 = ConvBlock(256, 256)
        self.conv19 = ConvBlock(128, 128)
        self.conv20 = ConvBlock(128, 128)
        self.conv21 = ConvBlock(64, 64)
        self.conv_61 = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.conv22 = ConvBlock(544, 512)

        self.conv23 = ConvBlock(512, 1024)
        self.conv24 = ConvBlock(512, 512)
        self.conv25 = ConvBlock(256, 256)
        self.conv26 = ConvBlock(256, 256)
        self.conv27 = ConvBlock(128, 128)
        self.conv28 = ConvBlock(128, 128)
        self.conv29 = ConvBlock(64, 64)

        self.up1 = UpBlock(2880, 512)
        self.up2 = UpBlock(1536, 512)
        self.up3 = UpBlock(1024, 256)
        self.up4 = UpBlock(768, 256)
        self.up5 = UpBlock(512, 128)
        self.up6 = UpBlock(384, 128)
        self.up7 = UpBlock(256, 64)
        self.conv30 = ConvBlock(256, 256)
        self.conv31 = ConvBlock(256, 256)
        self.conv32 = ConvBlock(256, 256)
        self.conv33 = ConvBlock(128, 128)
        self.conv34 = ConvBlock(128, 128)
        self.conv35 = ConvBlock(128, 128)
        self.conv36 = ConvBlock(128, 128)
        self.conv37 = ConvBlock(128, 128)
        self.conv38 = ConvBlock(128, 128)
        self.conv39 = ConvBlock(128, 128)
        self.conv40 = ConvBlock(64, 64)
        self.conv41 = ConvBlock(64, 64)
        self.conv42 = ConvBlock(64, 64)
        self.conv43 = ConvBlock(64, 64)
        self.conv44 = ConvBlock(64, 64)


        self.out_conv = nn.Conv2d(64, n_class, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)      # [B, 3, 384, 384] -> [B, 64, 384, 384]

        pool1 = self.pool1(conv1)  # [B, 64, 384, 384] -> [B, 64, 192, 192]
        conv2 = self.conv2(pool1)  # [B, 64, 192, 192] -> [B, 128, 192, 192]

        pool21 = self.pool2(conv1)  # [B, 64, 384, 384] -> [B, 64, 96, 96]
        conv31 = self.conv3(pool21)  # [B, 64, 96, 96] -> [B, 64, 96, 96]
        pool2 = self.pool1(conv2)   # [B, 128, 192, 192] -> [B, 128, 96, 96]
        conv3 = self.conv_21(pool21) # [B, 64, 96, 96] -> [B, 32, 96, 96] 普通卷积
        conv3 = self.bn1(conv3)
        conv3 = self.lr1(conv3)
        conv3 = torch.cat((conv3, pool2), dim=1)  # [B, 160, 96, 96]
        conv3 = self.conv4(conv3)  # [B, 160, 96, 96] -> [B, 128, 96, 96]

        pool3 = self.pool1(conv3)  # [B, 128, 96, 96] -> [B, 128, 48, 48]
        conv4 = self.conv5(pool3)  # [B, 128, 48, 48] -> [B, 256, 48, 48]
        pool31 = self.pool2(conv2) # [B, 128, 192, 192] -> [B, 128, 48, 48]
        conv41 = self.conv6(pool31) # [B, 128, 48, 48] -> [B, 128, 48, 48]
        pool32 = self.pool1(conv31) # [B, 64, 96, 96] -> [B, 64, 48, 48]
        conv42 = self.conv7(pool32) # [B, 64, 48, 48] -> [B, 64, 48, 48]

        pool41 = self.pool2(conv3) # [B, 128, 96, 96] -> [B, 128, 24, 24]
        conv51 = self.conv8(pool41)  # [B, 128, 24, 24] -> [B, 128, 24, 24]
        pool42 = self.pool1(conv41) # [B, 128, 48, 48] -> [B, 128, 24, 24]
        conv52 = self.conv9(pool42) # [B, 128, 24, 24] -> [B, 128, 24, 24]
        pool43 = self.pool1(conv42) # [B, 64, 48, 48] -> [B, 64, 24, 24]
        conv53 = self.conv10(pool43) # [B, 64, 24, 24] -> [B, 64, 24, 24]
        pool4 = self.pool1(conv4) # [B, 256, 48, 48] -> [B, 256, 24, 24]
        conv5 = self.conv_41(pool41) # [B, 128, 24, 24] -> [B, 32, 24, 24] 普通卷积
        conv5 = self.bn2(conv5)
        conv5 = self.lr2(conv5)
        conv5 = torch.cat((conv5, pool4), dim=1)  # [B, 288, 24, 24]
        conv5 = self.conv11(conv5) # [B, 288, 24, 24] -> [B, 256, 24, 24]

        pool5 = self.pool1(conv5) # [B, 256, 24, 24] -> [B, 256, 12, 12]
        conv6 = self.conv12(pool5) # [B, 256, 12, 12] -> [B, 512, 12, 12]
        pool51 = self.pool2(conv4) # [B, 256, 48, 48] -> [B, 256, 12, 12]
        conv61 = self.conv13(pool51)  # [B, 256, 12, 12] -> [B, 256, 12, 12]
        pool52 = self.pool1(conv51) # [B, 128, 24, 24] -> [B, 128, 12, 12]
        conv62 = self.conv14(pool52) # [B, 128, 12, 12] -> [B, 128, 12, 12]
        pool53 = self.pool1(conv52) # [B, 128, 24, 24] -> [B, 128, 12, 12]
        conv63 = self.conv15(pool53) # [B, 128, 12, 12] -> [B, 128, 12, 12]
        pool54 = self.pool1(conv53) # [B, 64, 24, 24] -> [B, 64, 12, 12]
        conv64 = self.conv16(pool54) # [B, 64, 12, 12] -> [B, 64, 12, 12]

        pool61 = self.pool2(conv5)  # [B, 256, 24, 24] -> [B, 256, 6, 6]
        conv71 = self.conv17(pool61) # [B, 256, 6, 6] -> [B, 256, 6, 6]
        pool62 = self.pool1(conv61) # [B, 256, 12, 12] -> [B, 256, 6, 6]
        conv72 = self.conv18(pool62) # [B, 256, 6, 6] -> [B, 256, 6, 6]
        pool63 = self.pool1(conv62) # [B, 128, 12, 12] -> [B, 128, 6, 6]
        conv73 = self.conv19(pool63) # [B, 128, 6, 6] -> [B, 128, 6, 6]
        pool64 = self.pool1(conv63) # [B, 128, 12, 12] -> [B, 128, 6, 6]
        conv74 = self.conv20(pool64) # [B, 128, 6, 6] -> [B, 128, 6, 6]
        pool65 = self.pool1(conv64) # [B, 64, 12, 12] -> [B, 64, 6, 6]
        conv75 = self.conv21(pool65) # [B, 64, 6, 6] -> [B, 64, 6, 6]
        pool6 = self.pool1(conv6) # [B, 512, 12, 12] -> [B, 512, 6, 6]
        conv7 = self.conv_61(pool61) # [B, 256, 6, 6] -> [B, 32, 6, 6]
        conv7 = self.bn3(conv7)
        conv7 = self.lr3(conv7)
        conv7 = torch.cat((conv7, pool6), dim=1) # [B, 544, 6, 6]
        conv7 = self.conv22(conv7) # [B, 544, 6, 6] -> # [B, 512, 6, 6]

        pool7 = self.pool1(conv7)  # [B, 512, 6, 6] -> [B, 512, 3, 3]
        conv8 = self.conv23(pool7) # [B, 512, 3, 3] -> [B, 1024, 3, 3]
        pool71 = self.pool2(conv6) # [B, 512, 12, 12] -> [B, 512, 3, 3]
        conv81 = self.conv24(pool71) # [B, 512, 3, 3] -> [B, 512, 3, 3]
        pool72 = self.pool1(conv71) # [B, 256, 6, 6] -> [B, 256, 3, 3]
        conv82 = self.conv25(pool72) # [B, 256, 3, 3] -> [B, 256, 3, 3]
        pool73 = self.pool1(conv72) # [B, 256, 6, 6] -> [B, 256, 3, 3]
        conv83 = self.conv26(pool73) # [B, 256, 3, 3] -> [B, 256, 3, 3]
        pool74 = self.pool1(conv73) # [B, 128, 6, 6] -> [B, 128, 3, 3]
        conv84 = self.conv27(pool74) # [B, 128, 3, 3] -> [B, 128, 3, 3]
        pool75 = self.pool1(conv74) # [B, 128, 6, 6] -> [B, 128, 3, 3]
        conv85 = self.conv28(pool75) # [B, 128, 3, 3] -> [B, 128, 3, 3]
        pool76 = self.pool1(conv75) # [B, 64, 6, 6] -> [B, 64, 3, 3]
        conv86 = self.conv29(pool76) # [B, 64, 3, 3] -> [B, 64, 3, 3]

        # 6 -> 2880
        concatenate1 = torch.cat((conv8, conv81, conv82, conv83, conv84, conv85, conv86), dim=1)  # [2368, 3, 3]
        up1 = self.up1(concatenate1, conv7)  # [2880, 3, 3] -> [512, 6, 6]
        # 12
        updata01 = F.interpolate(conv81, scale_factor=4, mode='bilinear') # [B, 512, 3, 3] -> [B, 512, 12, 12]
        concatenate2 = torch.cat((conv6, updata01), dim=1)  # [B, 1024, 12, 12]
        up2 = self.up2(up1, concatenate2)  # [B, 1536, 12, 12] -> [512, 12, 12]
        # 24
        updata2 = F.interpolate(conv82, scale_factor=2, mode='bilinear') # [B, 256, 3, 3] -> [B, 256, 6, 6]
        updata2 = self.conv30(updata2) # [B, 256, 6, 6] -> [B, 256, 6, 6]
        updata3 = F.interpolate(updata2, scale_factor=4, mode='bilinear') # [B, 256, 6, 6] -> [B, 256, 24, 24]
        concatenate3 = torch.cat((conv5, updata3), dim=1)  # [B, 512, 24, 24]
        up3 = self.up3(up2, concatenate3)  # [B, 1024, 24, 24] -> # [B, 256, 24, 24]
        # 48
        updata4 = F.interpolate(conv83, scale_factor=2, mode='bilinear') # [B, 256, 3, 3] -> [B, 256, 6, 6]
        updata4 = self.conv31(updata4) # [B, 256, 6, 6] -> [B, 256, 6, 6]
        updata4 = F.interpolate(updata4, scale_factor=2, mode='bilinear') # [B, 256, 6, 6] -> [B, 256, 12, 12]
        updata4 = self.conv32(updata4) # [B, 256, 12, 12] -> [B, 256, 12, 12]
        updata5 = F.interpolate(updata4, scale_factor=4, mode='bilinear') # [B, 256, 12, 12] -> [B, 256, 48, 48]
        concatenate4 = torch.cat((conv4, updata5), dim=1)  # [B, 512, 48, 48]
        up4 = self.up4(up3, concatenate4) # [B, 768, 48, 48] -> [B, 256, 48, 48]
        # 96
        updata6 = F.interpolate(conv84, scale_factor=2, mode='bilinear') # [B, 128, 3, 3] -> [B, 128, 6, 6]
        updata6 = self.conv33(updata6)  # [B, 128, 6, 6] -> [B, 128, 6, 6]
        updata6 = F.interpolate(updata6, scale_factor=2, mode='bilinear') # [B, 128, 6, 6] -> [B, 128, 12, 12]
        updata6 = self.conv34(updata6) # [B, 128, 12, 12] -> [B, 128, 12, 12]
        updata6 = F.interpolate(updata6, scale_factor=2, mode='bilinear') # [B, 128, 12, 12] -> [B, 128, 24, 24]
        updata6 = self.conv35(updata6) # [B, 128, 24, 24] -> [B, 128, 24, 24]
        updata7 = F.interpolate(updata6, scale_factor=4, mode='bilinear') # [B, 128, 24, 24] -> [B, 128, 96, 96]
        concatenate5 = torch.cat((conv3, updata7), dim=1) # [B, 256, 96, 96]
        up5 = self.up5(up4, concatenate5) # [B, 512, 96, 96] -> [B, 128, 96, 96]
        # 192
        updata8 = F.interpolate(conv85, scale_factor=2, mode='bilinear')  # [B, 128, 3, 3] -> [B, 128, 6, 6]
        updata8 = self.conv36(updata8)   # [B, 128, 6, 6] -> [B, 128, 6, 6]
        updata8 = F.interpolate(updata8, scale_factor=2, mode='bilinear') # [B, 128, 6, 6] -> [B, 128, 12, 12]
        updata8 = self.conv37(updata8)  # [B, 128, 12, 12] -> [B, 128, 12, 12]
        updata8 = F.interpolate(updata8, scale_factor=2, mode='bilinear')  # [B, 128, 12, 12] -> [B, 128, 24, 24]
        updata8 = self.conv38(updata8)   # [B, 128, 24, 24] -> [B, 128, 24, 24]
        updata8 = F.interpolate(updata8, scale_factor=2, mode='bilinear')  # [B, 128, 24, 24] -> [B, 128, 48, 48]
        updata8 = self.conv39(updata8)   # [B, 128, 48, 48] -> [B, 128, 48, 48]
        updata9 = F.interpolate(updata8, scale_factor=4, mode='bilinear')   # [B, 128, 48, 48] -> [B, 128, 192, 192]
        concatenate6 = torch.cat((conv2, updata9), dim=1)  # [B, 256, 192, 192]
        up6 = self.up6(up5, concatenate6)   # [B, 384, 192, 192]  -> [B, 128, 192, 192]
        # 384
        updata10 = F.interpolate(conv86, scale_factor=2, mode='bilinear')  # [B, 64, 3, 3] -> [B, 64, 6, 6]
        updata10 = self.conv40(updata10)   # [B, 64, 6, 6] -> [B, 64, 6, 6]
        updata10 = F.interpolate(updata10, scale_factor=2, mode='bilinear')  # [B, 64, 6, 6] -> [B, 64, 12, 12]
        updata10 = self.conv41(updata10)   # [B, 64, 12, 12] -> [B, 64, 12, 12]
        updata10 = F.interpolate(updata10, scale_factor=2, mode='bilinear')   # [B, 64, 12, 12] -> [B, 64, 24, 24]
        updata10 = self.conv42(updata10)   # [B, 64, 24, 24] -> [B, 64, 24, 24]
        updata10 = F.interpolate(updata10, scale_factor=2, mode='bilinear')  # [B, 64, 24, 24] -> [B, 64, 48, 48]
        updata10 = self.conv43(updata10)   # [B, 64, 48, 48] -> [B, 64, 48, 48]
        updata10 = F.interpolate(updata10, scale_factor=2, mode='bilinear')  # [B, 64, 48, 48] -> [B, 64, 96, 96]
        updata10 = self.conv44(updata10)   # [B, 64, 96, 96] -> [B, 64, 96, 96]
        updata11 = F.interpolate(updata10, scale_factor=4, mode='bilinear')  # [B, 64, 96, 96] -> [B, 64, 384, 384]
        concatenate7 = torch.cat((conv1, updata11), dim=1)   # [B, 128, 384, 384]
        up7 = self.up7(up6, concatenate7) # [B, 256, 192, 192]  -> [B, 64, 384, 384]

        out = self.out_conv(up7)  # [B, 64, 384, 384] -> [B, 9, 384, 384]
        out = self.sigmoid(out)
        return out

if __name__ == '__main__':
    x = torch.randn(1, 3, 384, 384).cuda()
    net = DeepUNet7(n_class=9).cuda()
    out = net(x)
    print(out.shape)


