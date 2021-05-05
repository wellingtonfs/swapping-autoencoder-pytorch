from models.networks import BaseNetwork
from models.networks.stylegan2_layers import ResBlock
import torch.nn as nn
import torch
import util

class CondPoseEncoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__(opt)
        #(W−F+2P)/S + 1
        self.opt = opt

        self.add_module(
            "conv_layers",
            nn.Sequential(
                #Show(),
                nn.Conv2d(in_channels=3+self.opt.num_layers_pose, out_channels=128, kernel_size=7, stride=2, padding=1), #128x126x126
                nn.LeakyReLU(0.01),
                #Show(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), #256x63x63
                nn.ReLU(),
                nn.BatchNorm2d(256),
                #Show(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1), #512x32x32
                nn.ReLU(),
                nn.BatchNorm2d(512),
                #Show(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1), #512x16x16
                nn.ReLU(),
                nn.BatchNorm2d(512),
                #Show(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1), #512x8x8
                nn.ReLU(),
                nn.BatchNorm2d(512),
                #Show(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1), #512x4x4
                nn.ReLU(),
                nn.BatchNorm2d(512)
                #Show()
            )
        )

        blur_kernel = [1, 2, 1] if self.opt.use_antialias else [1]

        self.add_module(
            "ResidualConv",
            nn.Sequential(
                ResBlock(512, 512, pad=1, downsample=False), #512x4x4
                ResBlock(512, 512, pad=1, downsample=False), #512x4x4
                ResBlock(512, 512, blur_kernel, reflection_pad=True), #512x2x2
                ResBlock(512, 512, blur_kernel, reflection_pad=False) #512x1x1
            )
        )

        self.add_module("fc_u", nn.Linear(in_features=512*1*1, out_features=self.opt.global_code_ch))
        self.add_module("fc_var", nn.Linear(in_features=512*1*1, out_features=self.opt.global_code_ch))
            
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)

        x = self.conv_layers(x) #512

        x = self.ResidualConv(x)

        x = x.view(x.size(0), -1)

        mu = self.fc_u(x)
        logvar = self.fc_var(x)
        
        return mu, logvar