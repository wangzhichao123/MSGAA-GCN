from gcn import Grapher
from pvtv2 import pvt_v2_b2
from gltb import GLTB
from fusion import Fusion

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class PvtGraphNet(nn.Module):
    def __init__(self, n_class=9):
        super(PvtGraphNet, self).__init__()
        self.n_class = n_class
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # ------------- GCN Decoder ---------------

        self.gcn4 = Grapher(512, 9, 1, 'mr', 'gelu', 'batch',
                            False, False, 0.2, 1, n=49, drop_path=0.0,
                            relative_pos=True)

        self.gcn3 = Grapher(320, 9, 1, 'mr', 'gelu', 'batch',
                            False, False, 0.2, 2, n=196, drop_path=0.0,
                            relative_pos=True)

        self.gcn2 = Grapher(128, 9, 4, 'mr', 'gelu', 'batch',
                            False, False, 0.2, 4, n=784, drop_path=0.0,
                            relative_pos=True)

        self.gcn1 = Grapher(64, 9, 3, 'mr', 'gelu', 'batch',
                            False, False, 0.2, 8, n=3136, drop_path=0.0,
                            relative_pos=True)

        self.gltb4 = GLTB(512)
        self.gltb3 = GLTB(320)
        self.gltb2 = GLTB(128)
        self.gltb1 = GLTB(64)

        self.mid4 = nn.Conv2d(512, 512, 1)
        self.mid3 = nn.Conv2d(320, 320, 1)
        self.mid2 = nn.Conv2d(128, 128, 1)
        self.mid1 = nn.Conv2d(64, 64, 1)

        self.fusion3 = Fusion(320, 512)
        self.fusion2 = Fusion(128, 320)
        self.fusion1 = Fusion(64, 128)


        self.pred4 = nn.Conv2d(512, self.n_class, 1)
        self.pred3 = nn.Conv2d(320, self.n_class, 1)
        self.pred2 = nn.Conv2d(128, self.n_class, 1)
        self.pred1 = nn.Conv2d(64, self.n_class, 1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        pvt = self.backbone(x)

        x1 = pvt[0]  # 64
        x2 = pvt[1]  # 128
        x3 = pvt[2]  # 320
        x4 = pvt[3]  # 512

        gcn4 = self.gcn4(x4)  # 512
        gltb4 = self.gltb4(gcn4)

        skip3 = self.fusion3(x3, gltb4)  # 320

        gcn3 = self.gcn3(skip3)  # 320
        gltb3 = self.gltb3(gcn3)

        skip2 = self.fusion2(x2, gltb3)  # 128

        gcn2 = self.gcn2(skip2)
        gltb2 = self.gltb2(gcn2)

        skip1 = self.fusion1(x1, gltb2)

        gcn1 = self.gcn1(skip1)
        gltb1 = self.gltb1(gcn1)

        prediction4 = self.pred4(gltb4)
        prediction3 = self.pred3(gltb3)
        prediction2 = self.pred2(gltb2)
        prediction1 = self.pred1(gltb1)

        prediction4 = F.interpolate(prediction4, scale_factor=32, mode='bilinear')
        prediction3 = F.interpolate(prediction3, scale_factor=16, mode='bilinear')
        prediction2 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        prediction1 = F.interpolate(prediction1, scale_factor=4, mode='bilinear')

        return prediction4, prediction3, prediction2, prediction1


if __name__ == '__main__':
    model = PvtGraphNet().cuda()
    x = torch.randn(4, 3, 352, 352).cuda()
    r1, r2, r3, r4 = model(x)
    print(r1.shape)
    print(r2.shape)
    print(r3.shape)
    print(r4.shape)
