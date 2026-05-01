import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v2


class PConv(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2)
        self.bn = nn.BatchNorm2d(out_ch)
        # self.act = nn.PReLU()
        self.act = nn.ReLU(inplace=True) #for quantization

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class RMU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pc5 = PConv(in_ch, out_ch, 5)
        self.pc1 = PConv(out_ch, out_ch, 1)
        self.pc3_1 = PConv(out_ch, out_ch, 3)
        self.pc3_2 = PConv(out_ch, out_ch, 3)

    def forward(self, x):
        f = self.pc1(self.pc5(x))
        f_res = self.pc3_1(f) + f
        up = F.interpolate(f_res, scale_factor=2, mode='nearest')
        out = self.pc3_2(up)
        return out
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v2(weights="IMAGENET1K_V1")
        self.features = backbone.features

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [2, 4, 7, 14]:  # multi-scale taps
                features.append(x)
        return features
    

class LightDepthNet(nn.Module):
    def __init__(self, max_depth=20.0):
        super().__init__()
        self.max_depth = max_depth # NYUv2 metric depth cap
        self.encoder = Encoder()

        self.rmu4 = RMU(160, 96)
        self.rmu3 = RMU(96, 64)
        self.rmu2 = RMU(64, 32)
        self.rmu1 = RMU(32, 16)

        self.skip3 = nn.Conv2d(64, 96, 1)
        self.skip2 = nn.Conv2d(32, 64, 1)
        self.skip1 = nn.Conv2d(24, 32, 1)

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)

        x = feats[-1]  # deepest feature (7x7)

        # Block 4
        x = self.rmu4(x) # scales up to 14x14
        x = x + self.skip3(feats[-2]) 
        
        # Block 3
        x = self.rmu3(x) # scales up to 28x28
        x = x + self.skip2(feats[-3])
        
        # Block 2
        x = self.rmu2(x) # scales up to 56x56
        x = x + self.skip1(feats[-4])
        
        # Block 1
        x = self.rmu1(x) # scales up to 112x112

        # Final upsample to match original 224x224 input
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        depth = self.final_conv(x)

        # Bound the output between 0 and max_depth (e.g., 10 meters)
        return torch.sigmoid(depth) * self.max_depth