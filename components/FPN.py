import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import CBAM

class FPNWithMCAndSpatialDropout(nn.Module):
    def __init__(self, backbone, out_channels=256, num_classes=1, dropout_prob=0.1, mc_dropout_prob=0.2):
        super(FPNWithMCAndSpatialDropout, self).__init__()
        self.initial_layers = nn.Sequential(
            backbone.conv1,  
            backbone.bn1,    
            backbone.relu,   
            backbone.maxpool 
        )

        self.layer1 = nn.Sequential(backbone.layer1, CBAM(256)) 
        self.layer2 = nn.Sequential(backbone.layer2, CBAM(512))  
        self.layer3 = nn.Sequential(backbone.layer3, CBAM(1024))  
        self.layer4 = nn.Sequential(backbone.layer4, CBAM(2048))  
        self.mc_dropout_layer2 = nn.Dropout2d(p=mc_dropout_prob)  
        self.mc_dropout_layer3 = nn.Dropout2d(p=mc_dropout_prob)  
        self.mc_dropout_layer4 = nn.Dropout2d(p=mc_dropout_prob)  

        self.lat_c5 = nn.Sequential(
            nn.Conv2d(2048, out_channels, kernel_size=1),
            nn.Dropout(p=mc_dropout_prob)
        )
        self.lat_c4 = nn.Sequential(
            nn.Conv2d(1024, out_channels, kernel_size=1),
            nn.Dropout(p=mc_dropout_prob)
        )
        self.lat_c3 = nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=1),
            nn.Dropout(p=mc_dropout_prob)
        )
        self.lat_c2 = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=1),
            nn.Dropout(p=mc_dropout_prob)
        )
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_dropout = nn.Dropout2d(p=dropout_prob)

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.initial_layers(x)
        c2 = self.layer1(x)  # 1/4 resolution
        c2 = self.mc_dropout_layer2(c2) 

        c3 = self.layer2(c2)  # 1/8 resolution
        c3 = self.mc_dropout_layer3(c3)  

        c4 = self.layer3(c3)  # 1/16 resolution
        c4 = self.mc_dropout_layer4(c4) 

        c5 = self.layer4(c4)  # 1/32 resolution

        p5 = self.lat_c5(c5)  
        p4 = self.smooth_p4(self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest"))  
        p3 = self.smooth_p3(self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest"))  
        p2 = self.smooth_p2(self.lat_c2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest"))  
        segmentation_output = self.segmentation_head(p2)

        return segmentation_output
