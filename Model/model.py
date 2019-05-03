

import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function



def cifar_shakeshake26(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet32x32(ShakeShakeBlock,
                        layers=[4, 4, 4],
                        channels=96, **kwargs)
    return model



class ResNet32x32(nn.Module):
    def __init__(self, block, layers, channels, groups=1, num_classes=10, supervised = True):
        super().__init__()
        assert len(layers) == 3

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, channels, groups, layers[0], sample_mode='basic')#16->96
        self.layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=2) #96 -> 96*2
        self.layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=2) # 96*2 -> 96*3

        self.avgpool = nn.AvgPool2d(8)
        if supervised:
            self.fc = nn.Linear(block.out_channels(channels * 4, groups), num_classes)
        block = ShakeShakeDecBlock
        self.layerD3 = self._make_layer(block, channels * 2, groups, layers[2], stride=2, sample_mode='deconv')
        self.layerD2 = self._make_layer(block, channels, groups, layers[1], stride=2, sample_mode='deconv')
        self.layerD1 = self._make_layer(block, 16, groups, layers[0])

        self.conv2 = nn.Sequential(
                        nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(3),
                        nn.Tanh())

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1, sample_mode='basic'):
        sample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if sample_mode == 'basic' or stride == 1:
                sample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif sample_mode == 'shift_conv':
                sample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            elif sample_mode == 'deconv':
                sample = nn.Sequential(
                    nn.ConvTranspose2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=2, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, sample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, x, superivsed):
        h1 = self.conv1(x)
        h2 = self.layer1(h1)
        h3 = self.layer2(h2)
        h = self.layer3(h3)
        if superivsed:
            x = self.avgpool(h)
            x = x.view(x.size(0), -1)
            hat_y = self.fc(x)
        else:
            hat_y = None

        hat_h3 = self.layerD3(h)
        hat_h2 = self.layerD2(hat_h3)
        hat_h1 = self.layerD1(hat_h2)
        hat_x = self.conv2(hat_h1)
        return hat_x, h1, h2, h3, hat_h1, hat_h2, hat_h3, h, hat_y


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def Dconv3x3(in_planes, out_planes, stride=2):
    "3x3 convolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                     padding=0, bias=False)


class BottleneckBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        if groups > 1:
            return 2 * planes
        else:
            return 4 * planes

    def __init__(self, inplanes, planes, groups, stride=1, sample=None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv_a1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn_a2 = nn.BatchNorm2d(planes)
        self.conv_a3 = nn.Conv2d(planes, self.out_channels(
            planes, groups), kernel_size=1, bias=False)
        self.bn_a3 = nn.BatchNorm2d(self.out_channels(planes, groups))

        self.sample = sample
        self.stride = stride

    def forward(self, x):
        a, residual = x, x

        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = self.relu(a)
        a = self.conv_a2(a)
        a = self.bn_a2(a)
        a = self.relu(a)
        a = self.conv_a3(a)
        a = self.bn_a3(a)

        if self.sample is not None:
            residual = self.downsample(residual)

        return self.relu(residual + a)


class ShakeShakeBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, sample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.conv_b1 = conv3x3(inplanes, planes, stride)
        self.bn_b1 = nn.BatchNorm2d(planes)
        self.conv_b2 = conv3x3(planes, planes)
        self.bn_b2 = nn.BatchNorm2d(planes)

        self.sample = sample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x

        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)

        ab = shake(a, b, training=self.training)

        if self.sample is not None:
            residual = self.sample(x)
        # print("A :  ", a.size(), "B :  ", b.size(), "AB :  ", ab.size(), "res :  ", residual.size())

        return residual + ab

class ShakeShakeDecBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, sample=None):
        super().__init__()
        assert groups == 1
        if stride == 2:
            self.conv_a1 = Dconv3x3(inplanes, planes, stride)
        else:
            self.conv_a1 = conv3x3(inplanes, planes, stride)

        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        if stride ==2:
            self.conv_b1 = Dconv3x3(inplanes, planes, stride)
        else:
            self.conv_b1 = conv3x3(inplanes, planes, stride)

        self.bn_b1 = nn.BatchNorm2d(planes)
        self.conv_b2 = conv3x3(planes, planes)
        self.bn_b2 = nn.BatchNorm2d(planes)

        self.sample = sample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x

        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)

        ab = shake(a, b, training=self.training)

        if self.sample is not None:
            residual = self.sample(x)

        # print("A :  ", a.size() , "B :  ", b.size(), "AB :  ", ab.size(), "res :  ", residual.size())
        return residual + ab


class Shake(Function):
    @classmethod
    def forward(cls, ctx, inp1, inp2, training):
        assert inp1.size() == inp2.size()
        gate_size = [inp1.size()[0], *itertools.repeat(1, inp1.dim() - 1)]
        gate = inp1.new(*gate_size)
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)
        return inp1 * gate + inp2 * (1. - gate)

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_inp1 = grad_inp2 = grad_training = None
        gate_size = [grad_output.size()[0], *itertools.repeat(1,
                                                              grad_output.dim() - 1)]
        gate = Variable(grad_output.data.new(*gate_size).uniform_(0, 1))
        if ctx.needs_input_grad[0]:
            grad_inp1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_inp2 = grad_output * (1 - gate)
        assert not ctx.needs_input_grad[2]
        return grad_inp1, grad_inp2, grad_training


def shake(inp1, inp2, training=False):
    return Shake.apply(inp1, inp2, training)


class ShiftConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=2 * in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              groups=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat((x[:, :, 0::2, 0::2],
                       x[:, :, 1::2, 1::2]), dim=1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
