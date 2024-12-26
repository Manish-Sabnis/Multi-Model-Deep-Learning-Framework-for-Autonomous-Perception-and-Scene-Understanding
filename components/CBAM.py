import torch
import torch.nn as nn
import torch.optim as optim



class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
       
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
       
        avg_out = self.channel_attention[0](x)
        max_out = self.channel_attention[1](x)
        out = self.channel_attention[2:](avg_out + max_out)
        x = x * out

       
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.spatial_attention(out)
        x = x * out
        return x
