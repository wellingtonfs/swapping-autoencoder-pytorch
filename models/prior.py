import torch
import torch.nn as nn

class Prior(nn.Module):
    def __init__(self, dim_y=3):
        super().__init__()
        #(W−F+2P)/S + 1
        #(64-4+2)/2 + 1

        self.conv_layers = nn.Sequential(
            #Show(),
            nn.Conv2d(in_channels=dim_y, out_channels=128, kernel_size=4, stride=2, padding=1), #128x32x32
            nn.LeakyReLU(0.2),
            #Show(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), #256x16x16
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            #Show(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), #512x8x8
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            #Show(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1), #1024x4x4
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(1024),
            #Show(),
            nn.Conv2d(in_channels=1024, out_channels=100, kernel_size=4, stride=1, padding=0), #100x1x1
            nn.Sigmoid()
        )

        self.fc_u = nn.Linear(in_features=100*1*1, out_features=latent_dims)
        self.fc_var = nn.Linear(in_features=100*1*1, out_features=latent_dims)

        self.sigmoid = nn.Sigmoid()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, y):
        y = self.conv_layers(y)
        y = y.view(y.size(0), -1)
        mu = self.fc_u(y)
        var = self.fc_var(y)
        return mu, var