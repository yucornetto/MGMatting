from   networks.decoders.resnet_dec import ResNet_D_Dec
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResShortCut_D_Dec(ResNet_D_Dec):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False):
        super(ResShortCut_D_Dec, self).__init__(block, layers, norm_layer, large_kernel,
                                                late_downsample=late_downsample)

    def forward(self, x, mid_fea):
        ret = {}
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        x_os8 = self.refine_OS8(x)

        x = self.layer3(x) + fea3
        x_os4 = self.refine_OS4(x)

        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x_os1 = self.refine_OS1(x)
        
        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        return ret

