import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet as models

class PPM(nn.Module):
    def __init__(self, in_dim, out_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU()
                )
            )
        self.features = nn.ModuleList(self.features)


    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            # Tensor shape [batch, c, h, w]
            # h, w = x.size(2), x.size(3)
            out.append(F.interpolate(f(x), size=x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, dim=1)


'''pyramid sence parsing network'''
class PSPNet(nn.Module):
    def __init__(self, layers=50, nclass = 21, bins=(1,2,3,6), dropout=0.1, zoom_factor=8, pretrained=None):
        super(PSPNet, self).__init__()
        assert layers in [18, 34, 50, 101, 152]
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor

        # backbone ResNet
        fea_dim = 2048      # ResNet输出map维度，用于后续PSP module
        mid_dim = 1024      # Resnet block4 输出的feature map维度，用于训练时的aux loss

        if layers == 18:
            resnet = models.resnet18()
            fea_dim = 512
            mid_dim = 256
        elif layers == 34:
            resnet = models.resnet34()
            fea_dim = 512
            mid_dim = 256
        elif layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101()
        else:
            resnet = models.resnet152()
        '''
        if pretrained is not None:
            print("Loading pretrained model of resnet-{} from {}...".format(layers, pretrained))
            resnet.load_state_dict(torch.load(pretrained), strict=False)
            print("Loading successfully!")
        '''

        assert fea_dim % len(bins) == 0

        # 设置空洞卷积
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu,resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)


        # Add PPM
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)

        # Add final Prediction
        self.final = nn.Sequential(
            nn.Conv2d(2*fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Conv2d(512, nclass, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(mid_dim, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, nclass, kernel_size=1)
            )

    def forward(self, x):
        x_size = x.size()
        # zoom factor = 8
        if self.training:
            assert  x_size[2] % 8 == 0 and x_size[3] % 8 == 0
        h = x_size[2]
        w = x_size[3]

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        mid = self.layer3(x)    # For Auxiliary Loss 2
        x = self.layer4(mid)

        x = self.ppm(x)
        x = self.final(x)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h,w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(mid)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h,w), mode='bilinear', align_corners=True)
            return x, aux
        else:
            return x