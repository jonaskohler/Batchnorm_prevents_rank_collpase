import math

import torch.nn as nn
import torch.nn.init as init

import my_bn as my_bn #in this model we can modify BN, e.g. do only centering or only std adaption
import torch
import IPython

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features=None, seed=None):
        super(VGG, self).__init__()
        seed=torch.LongTensor(1).random_(0, 500)  ################ manually changing seed
        if seed is not None:
            torch.manual_seed(seed)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights with He init
        if False:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    #m.weight.data.uniform_(-math.sqrt(6./n),+math.sqrt(6./n))
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x_final = self.classifier(x)
        return x, x_final


def make_layers(cfg, batch_norm=False,centering=False,normalize_std=False,fixed_std=False, weightnorm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, my_bn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif centering:
            	layers += [conv2d, my_bn.BatchNorm2d(v,centering=True), nn.ReLU(inplace=True)]
            elif normalize_std:
            	layers += [conv2d, my_bn.BatchNorm2d(v,normalize_std=True), nn.ReLU(inplace=True)]
            elif fixed_std:
            	layers += [conv2d, my_bn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif weightnorm:
                layers += [nn.utils.weight_norm(conv2d), nn.ReLU(inplace=True)]

            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_centering():
	return VGG(make_layers(cfg['E'], centering=True))

def vgg19_std():
	return VGG(make_layers(cfg['E'], normalize_std=True))

def vgg19_fixed_std():
	return VGG(make_layers(cfg['E'], fixed_std=True))


def vgg19_wn():
    return VGG(make_layers(cfg['E'], weightnorm=True))




def vgg19():
    """VGG 19-layer model (configuration "E")"""
    #return VGG(make_layers(cfg['E']))
    return VGG_manual()


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))