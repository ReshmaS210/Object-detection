import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import swin_v2_t

from modules import Conv, HFAM, AGPU


class SwinV2_AGPU_HFAM_FPN(nn.Module):
    """
    V3 Changes:
    - Removed Dropout2d (FIX 6)
    - Added C2 extraction for finer-grained features (FIX 3 — 4 FPN levels)
    - Added P6 via max-pool for extra scale
    """
    def __init__(self, out_channels=256):
        super().__init__()
        swin = swin_v2_t(weights=None)
        self.features = swin.features
        self.norm = swin.norm

        # Swin V2-T channel dims: Stage1=96, Stage2=192, Stage3=384, Stage4=768
        in_ch = [96, 192, 384, 768]

        # Lateral projections for 4 scales
        self.proj_c2 = nn.Conv2d(in_ch[0], out_channels, 1, bias=False)
        self.proj_c3 = nn.Conv2d(in_ch[1], out_channels, 1, bias=False)
        self.proj_c4 = nn.Conv2d(in_ch[2], out_channels, 1, bias=False)
        self.proj_c5 = nn.Conv2d(in_ch[3], out_channels, 1, bias=False)
        for m in [self.proj_c2, self.proj_c3, self.proj_c4, self.proj_c5]:
            nn.init.kaiming_uniform_(m.weight, a=1)

        # AGPU + HFAM for top-down fusion
        self.agpu1 = AGPU([out_channels, out_channels])
        self.hfam1 = HFAM(out_channels*2, out_channels, n=3, shortcut=False, p=2, kernel_size=3)

        self.agpu2 = AGPU([out_channels, out_channels])
        self.hfam2 = HFAM(out_channels*2, out_channels, n=3, shortcut=False, p=2, kernel_size=3)

        self.agpu3 = AGPU([out_channels, out_channels])
        self.hfam3 = HFAM(out_channels*2, out_channels, n=3, shortcut=False, p=2, kernel_size=3)

        # Smoothing convs
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.kaiming_uniform_(self.smooth_p5.weight, a=1)
        nn.init.zeros_(self.smooth_p5.bias)

        self.out_channels = out_channels

    def forward(self, x):
        # Swin V2-T stages: features[0]=patch_embed, [1]=stage1, [2]=downsample, [3]=stage2, [4]=downsample, [5]=stage3, [6]=downsample, [7]=stage4
        x = self.features[0](x)  # patch embed
        x = self.features[1](x)  # stage 1
        c2 = x.permute(0, 3, 1, 2)  # (B, 96, H/4, W/4)

        x = self.features[2](x)  # downsample
        x = self.features[3](x)  # stage 2
        c3 = x.permute(0, 3, 1, 2)  # (B, 192, H/8, W/8)

        x = self.features[4](x)  # downsample
        x = self.features[5](x)  # stage 3
        c4 = x.permute(0, 3, 1, 2)  # (B, 384, H/16, W/16)

        x = self.features[6](x)  # downsample
        x = self.features[7](x)  # stage 4
        c5 = x.permute(0, 3, 1, 2)  # (B, 768, H/32, W/32)

        # Lateral projections
        p2, p3, p4, p5 = self.proj_c2(c2), self.proj_c3(c3), self.proj_c4(c4), self.proj_c5(c5)

        # Top-down AGPU+HFAM fusion
        out_4 = self.hfam1(self.agpu1([p4, p5]))       # fuse P5 into P4
        out_3 = self.hfam2(self.agpu2([p3, out_4]))     # fuse P4 into P3
        out_2 = self.hfam3(self.agpu3([p2, out_3]))     # fuse P3 into P2
        out_5 = self.smooth_p5(p5)

        return OrderedDict([('0', out_2), ('1', out_3), ('2', out_4), ('3', out_5)])
