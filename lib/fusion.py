import torch
import torch.nn as nn

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=15, stride=1, padding=7)  ####kernel_size=7
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		    nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		    nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.up(x)
        return x
class Fusion(nn.Module):

    def __init__(self, en_channel, de_channel):
        super(Fusion, self).__init__()
        self.conv1x1_en = nn.Sequential(nn.Conv2d(en_channel, en_channel, kernel_size=1),
                                        nn.BatchNorm2d(en_channel),
                                        nn.ReLU(inplace=True))
        self.conv1x1_de = up_conv(de_channel, en_channel)

        self.cbam_en = CBAM(en_channel)
        self.cbam_de = CBAM(en_channel)

        self.conv = nn.Sequential(nn.Conv2d(en_channel, en_channel, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(en_channel),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(en_channel, en_channel, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(en_channel),
                                  nn.LeakyReLU(inplace=True))


    def forward(self, en_feature, de_feature):
        en1 = self.conv1x1_en(en_feature)
        de1 = self.conv1x1_de(de_feature)

        att_en = self.cbam_en(en1)
        att_de = self.cbam_de(de1)

        att_ed = att_en + att_de
        att_ed_1 = att_ed

        out = self.conv(att_ed)

        return out

if __name__ == '__main__':
    en = torch.rand(4,320,22,22).cuda()
    de = torch.rand(4,512,11,11).cuda()
    f = Fusion(320, 512).cuda()
    print(f(en,de).shape)