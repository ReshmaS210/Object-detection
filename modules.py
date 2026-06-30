import math
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training: return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    r = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    r.floor_()
    return x.div(keep) * r

class DropPath(nn.Module):
    def __init__(self, p=None): super().__init__(); self.drop_prob = p
    def forward(self, x): return drop_path(x, self.drop_prob, self.training)

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=((3,3),(3,3)), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x): return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path_rate=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    def forward(self, x):
        inp = x; x = self.dwconv(x)
        x = self.act(self.f1(x)) * self.f2(x)
        return inp + self.drop_path(self.dwconv2(self.g(x)))

class MANet(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3,3),(3,3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2*self.c, dim_hid, 1, 1), DWConv(dim_hid, dim_hid, kernel_size, 1), Conv(dim_hid, self.c, 1, 1))
    def forward(self, x):
        y = self.cv_first(x); y0 = self.cv_block_1(y); y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1); y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv_final(torch.cat(y, 1))

class HFAM(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Star_Block(self.c) for _ in range(n))

class AGPU(nn.Module):
    def __init__(self, input_channel, gamma=2, bias=1):
        super().__init__()
        c1, c2 = input_channel
        self.input_channel1, self.input_channel2 = c1, c2
        self.avg1, self.avg2 = nn.AdaptiveAvgPool2d(1), nn.AdaptiveAvgPool2d(1)
        ks1 = int(abs((math.log(c1,2)+bias)/gamma)); ks1 = ks1 if ks1%2 else ks1+1
        ks2 = int(abs((math.log(c2,2)+bias)/gamma)); ks2 = ks2 if ks2%2 else ks2+1
        ks3 = int(abs((math.log(c1+c2,2)+bias)/gamma)); ks3 = ks3 if ks3%2 else ks3+1
        self.conv1 = nn.Conv1d(1,1,kernel_size=ks1,padding=(ks1-1)//2,bias=False)
        self.conv2 = nn.Conv1d(1,1,kernel_size=ks2,padding=(ks2-1)//2,bias=False)
        self.conv3 = nn.Conv1d(1,1,kernel_size=ks3,padding=(ks3-1)//2,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.ConvTranspose2d(c2, c1, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x):
        x1, x2 = x
        x1_ = self.conv1(self.avg1(x1).squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        x2_ = self.conv2(self.avg2(x2).squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        xm = self.sigmoid(self.conv3(torch.cat((x1_,x2_),dim=1).squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1))
        a1, a2 = torch.split(xm, [self.input_channel1, self.input_channel2], dim=1)
        return torch.cat([x1*a1, self.up(x2*a2)], dim=1)

print('EAMSF-DETR modules loaded.')
