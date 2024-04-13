import torch
import torch.nn as nn
from lib.MSGAT import MSGroupAgentAttention
from lib.Att import AA_kernel

class GLTB2(nn.Module):
    def __init__(self, in_c):
        super(GLTB2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.bn2 = nn.BatchNorm2d(in_c)
        
        self.aa_kernel_1 = AA_kernel(in_c, in_c)
        
        self.local1x1 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=1, padding=0), nn.BatchNorm2d(in_c))
        self.local3x3 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.BatchNorm2d(in_c))
        self.local5x5 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=5, padding=2), nn.BatchNorm2d(in_c))

        # 7x7 Depth-wise Conv
        self.dw_conv = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=7, padding=3, groups=in_c),
                                     nn.BatchNorm2d(in_c),
                                     nn.Conv2d(in_c, in_c, kernel_size=1))

        self.ffn = FFN(in_c, 4 * in_c, in_c) # 可调参数

    def forward(self, x):
        x_1 = x
        x = self.bn1(x)
        # ----- global ------
        g = self.aa_kernel_1(x)
        # # ------ local -----
        l1 = self.local1x1(x)
        l2 = self.local3x3(x)
        l3 = self.local5x5(x)
        l = l1 + l2 + l3              # torch.Size([4, 64, 11, 11])
        d1 = self.dw_conv(l+g)
        d = d1 + x_1
        d_1 = d
        d = self.bn2(d)
        d = self.ffn(d)
        out = d_1 + d

        return out

class GLTB1(nn.Module):
    def __init__(self, in_c, num_heads, num_patches, qkv_bias, sr_ratio, agent_num):
        super(GLTB1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.bn2 = nn.BatchNorm2d(in_c)
        self.sgaa = MSGroupAgentAttention(dim=in_c, num_heads=num_heads, num_patches=num_patches, qkv_bias=qkv_bias, sr_ratio=sr_ratio, agent_num=agent_num)

        self.local1x1 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=1, padding=0), nn.BatchNorm2d(in_c))
        self.local3x3 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.BatchNorm2d(in_c))
        self.local5x5 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=5, padding=2), nn.BatchNorm2d(in_c))

        # 7x7 Depth-wise Conv
        self.dw_conv = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=7, padding=3, groups=in_c),
                                     nn.BatchNorm2d(in_c),
                                     nn.Conv2d(in_c, in_c, kernel_size=1))

        self.ffn = FFN(in_c, 4 * in_c, in_c)

    def forward(self, x):
        x_1 = x
        x = self.bn1(x)
        # ----- global ------
        g = self.sgaa(x)
        # # ------ local -----
        l1 = self.local1x1(x)
        l2 = self.local3x3(x)
        l3 = self.local5x5(x)
        l = l1 + l2 + l3              # torch.Size([4, 64, 11, 11])
        d1 = self.dw_conv(l+g)
        d = d1 + x_1
        d_1 = d
        d = self.bn2(d)
        d = self.ffn(d)
        out = d_1 + d

        return out
        
class GLTB(nn.Module):
    def __init__(self, in_c):
        super(GLTB, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.bn2 = nn.BatchNorm2d(in_c)
        self.pam_attention_1_1 = PAM_CAM_Layer(in_c, True)
        self.cam_attention_1_1 = PAM_CAM_Layer(in_c, False)
        self.conv1_1 = nn.Conv2d(in_c, in_c, kernel_size=1, padding=0)

        self.local1x1 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=1, padding=0), nn.BatchNorm2d(in_c))
        self.local3x3 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.BatchNorm2d(in_c))
        self.local5x5 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=5, padding=2), nn.BatchNorm2d(in_c))

        # Depth-wise Conv
        self.dw_conv = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=7, padding=3, groups=in_c),
                                     nn.BatchNorm2d(in_c),
                                     nn.Conv2d(in_c, in_c, kernel_size=1, padding=0))

        self.ffn = FFN(in_c, 4 * in_c, in_c)

    def forward(self, x):
        x_1 = x
        x = self.bn1(x)
        # ----- global ------
        g1 = self.pam_attention_1_1(x)
        g2 = self.cam_attention_1_1(x)
        g = self.conv1_1(g1 + g2)     
        # # ------ local -----
        l1 = self.local1x1(x)
        l2 = self.local3x3(x)
        l3 = self.local5x5(x)
        l = l1 + l2 + l3              
        d1 = self.dw_conv(l+g)
        d = d1 + x_1
        d_1 = d
        d = self.bn2(d)
        d = self.ffn(d)
        out = d_1 + d

        return out


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super(FFN, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
        
class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 1, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, height, width = x.size()
        proj_query = self.query_conv(x).view(B, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        B, C, height, width = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(B, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(B, C, height, width)

        out = self.gamma * out + x
        return out

class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch, use_pam=True):
        super(PAM_CAM_Layer, self).__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            PAM_Module(in_ch) if use_pam else CAM_Module(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.attn(x)


class MultiConv(nn.Module):
    def __init__(self, in_ch, out_ch, attn=True):
        super(MultiConv, self).__init__()

        self.fuse_attn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Softmax2d() if attn else nn.PReLU()
        )

    def forward(self, x):
        return self.fuse_attn(x)

if __name__ == '__main__':
    x = torch.randn(4,64,11,11).cuda()
    gltb = GLTB(64).cuda()
    print(gltb(x).shape)
