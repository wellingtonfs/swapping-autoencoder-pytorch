import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_layers=3, dim_y=3):
        super().__init__()
        #(W−F+2P)/S + 1

        self.conv_layers = nn.Sequential(
            #Show(),
            nn.Conv2d(in_channels=in_layers+dim_y, out_channels=128, kernel_size=7, stride=2, padding=1), #128x30x30
            nn.LeakyReLU(0.01),
            #Show(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), #256x15x15
            nn.ReLU(),
            nn.BatchNorm2d(256),
            #Show(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1), #512x8x8
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #Show(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1), #512x4x4
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #Show(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1), #512x2x2
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #Show(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1), #512x1x1
            nn.ReLU(),
            nn.BatchNorm2d(512)
            #Show()
        )

        self.ResidualConv = [
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), #512x1x1
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), #512x1x1
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), #512x1x1
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1) #512x1x1
        ]

        self.fc_u = nn.Linear(in_features=512*1*1, out_features=latent_dims)
        self.fc_var = nn.Linear(in_features=512*1*1, out_features=latent_dims)

        self.sigmoid = nn.Sigmoid()

        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.conv_layers(x) #512

        for function in self.ResidualConv:
            identity = x

            if torch.cuda.is_available():
                function.cuda()

            x = function(x)
            x += identity

        x = self.sigmoid(x)

        x = x.view(x.size(0), -1)

        mu = self.fc_u(x)
        var = self.fc_var(x)
        return mu, var