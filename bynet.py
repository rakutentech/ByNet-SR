import torch
import torch.nn as nn
from math import sqrt
    
class Bypass(nn.Module):
    def __init__(self, conv_connection = False):
        super(Bypass, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv_connection = conv_connection
        self.conv64c_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv64c_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv64c_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv64c_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        conv_1 = self.relu(self.conv64c_1(x))
        conv_2 = self.relu(self.conv64c_2(conv_1))
        conv_3 = self.conv64c_3(conv_2)
        if self.conv_connection is False:
            output = torch.add(identity_data,conv_3)
        else:
            conv_connection_data = self.conv64c_4(identity_data)
            output = torch.add(conv_connection_data,conv_3)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            Bypass(conv_connection = False),
            Bypass(conv_connection = True),
            Bypass(conv_connection = False),
            Bypass(conv_connection = True),
            Bypass(conv_connection = False),
            Bypass(conv_connection = True),
            Bypass(conv_connection = False),
            Bypass(conv_connection = True),
            Bypass(conv_connection = False),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.scale = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

        self.scale.weight.data.fill_(0.1)

    def forward(self, x):
        residual = x
        out = self.features(x)
        out = self.scale(out)
        out = torch.add(out,residual)

        return out
