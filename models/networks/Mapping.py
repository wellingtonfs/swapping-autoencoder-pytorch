from models.networks import BaseNetwork
from models.networks.stylegan2_layers import ResBlock
import torch.nn as nn
import torch
import util

class CondPoseMapping(BaseNetwork):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

        self.add_module(
            "conv_layers",
            nn.Sequential(
                nn.Conv2d(in_channels=self.opt.num_layers_pose, out_channels=128, kernel_size=4, stride=2, padding=1), #128x128x128
                nn.LeakyReLU(0.2),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), #256x64x64
                nn.LeakyReLU(0.2),
                #nn.BatchNorm2d(256),

                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), #512x32x32
                nn.LeakyReLU(0.2),
                #nn.BatchNorm2d(512),

                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1), #512x16x16
                nn.LeakyReLU(0.2)
            )
        )

        self.add_module(
            "ResidualConv",
            nn.Sequential(
                ResBlock(512, 512, pad=1, downsample=False), #512x16x16
                ResBlock(512, 512, pad=1, downsample=False), #512x16x16
                ResBlock(512, 512, pad=1, downsample=False), #512x16x16
                ResBlock(512, 512, pad=1, downsample=False) #512x16x16
            )
        )

        self.add_module(
            "conv_out",
            nn.Conv2d(
                in_channels=512,
                out_channels=self.opt.spatial_code_ch,
                kernel_size=3, stride=1, padding=1), #8x16x16
        )

    def forward(self, y):
        out = self.conv_layers(y)

        out = self.ResidualConv(out)

        return util.normalize(self.conv_out(out))