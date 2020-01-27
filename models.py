import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Our batch shape for input x is (3, 224, 224)

expected input is (C, H, W), channels = 3 

h, w in [(1030, 1030), (570,570), (254,254), (600, 600)]
# h-3
# ((h-3)-3)/2
# ((((h-3)-3)/2) - 3)/2
# ((((((h-3)-3)/2) -3)/2) - 1)/2
# ((((((((h-3)-3)/2) -3)/2) - 1)/2)
# 4096(1024*2*2) input features, 1024 output features
"""


class Encoder(nn.Module):
    def __init__(self, ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, 64, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(256, 1024, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2*2*1024, 1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # Size changes from (1, 4096) to (1, 1024)
        x = self.fc1(x.view(-1, 2 * 2 * 1024))
        # Computes the activation of the first fully connected layer
        x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256*8*8, 1)
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=5)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=5)
        self.deconv4 = nn.ConvTranspose2d(128,  64, kernel_size=5)
        self.deconv5 = nn.ConvTranspose2d(64,    3, kernel_size=5)

    def forward(self, x):
        x = self.fc(x)
        x = self.deconv1(x.view(1, 256, 1, 1))
        x = self.deconv2(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x


class WE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = Encoder()
        self.we = WE()
        self.tx = nn.Linear(1024, 1024)
        self.decoder = Decoder()

    def forward(self, x1, x2, w=None):
        x1 = self.encoder1(x1)
        x2 = self.encoder1(x2)
        _w = self.we(w)
        x = torch.cat([x1, x2])
        x = torch.cat([x, _w])
        x = self.tx(x)
        # -->
        x = self.decoder(x)
        return x


class Encoder2(nn.Module):
    """Our batch shape for input x is (3, 224, 224) """

    def __init__(self):
        super().__init__()
        # Input channels = 3, output channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, )
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2,)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2)

        # 4096(1024*2*2) input features, 1024 output features
        self.fc1 = nn.Linear(2*2*1024, 1024)

    def forward(self, x1):
        return x1


class Decoder2():
    NUNM = 5

    def __init__(self):
        super().__init__()
        self.deconv0 = nn.ConvTranspose2d(NUMM, 512, kernel_size=4, stride=2)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.deconv6 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2)

    def forward(self, x1):
        return x1


class LeNet2():
    def __init__(self):
        super().__init__()
        self.encoder1 = Encoder2()
        self.decoder = Decoder2()

    def forward(self, x1, x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder1(x2)
        return x1.concat(x2)


class ConditionD(nn.Module):
    NUNM = 5

    def __init__(self):
        self.conv0 = nn.Conv2d(NUNM, 64, 4, stride=2)
        self.conv1 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2)
        self.conv3 = nn.Conv2d(256, 512, 4, stride=1)
        self.conv4 = nn.Conv2d(512, 1, 4, stride=1)

    def forward(self, x1, x2):
        return x1 - x2
