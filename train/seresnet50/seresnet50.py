from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import torch.nn as nn
from torch.utils import model_zoo

__all__ = ['SENet', 'se_resnet50']

pretrained_settings = {
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)    # 平均池化函数，1表示输出尺寸为1x1
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)          # ReLU激活函数，inplace表示经过激活函数输入是否变化
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()                # sigmoid激活函数

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    bottlenecks的基类, 由forward()方法实现
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, inplanes=128,
                 downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
        """
        参数
        block (nn.Module): Bottleneck class.比如SEResNetBottleneck
        layers (list of ints): 4层，每层残次块的数量 (layer1...layer4).
        groups (int): 每个bottleneck block中的3x3卷积个数
            - For SENet154: 64
            - For SE-ResNet models: 1
        reduction (int): Squeeze-and-Excitation modules的Reduction ratio，默认为16
        inplanes (int):  layer1输入通道的数量
            - For SENet154: 128
            - For SE-ResNet models: 64
        downsample_kernel_size (int): layer2, layer3 and layer4下采样卷积的 Kernel size
            - For SENet154: 3
            - For SE-ResNet models: 1
        downsample_padding (int): layer2, layer3 and layer4下采样卷积的Padding
            - For SENet154: 1
            - For SE-ResNet models: 0
        num_classes (int): 最后一层输出神经元个数，默认都是1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                          ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True)),
                          ('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True))]

        # To preserve compatibility with Caffe weights `ceil_mode=True` is used instead of `padding=1`.
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))   # output 64 * 56 * 56
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],  # [3, 4, 6, 3]中的3
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0      # layer 1 不会降尺寸。但是会改变通道。所以输出是256*56*56
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],  # [3, 4, 6, 3]中的4
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding   # layer 2降尺寸。因为stride =2要进行降采样。输出就是 512 * 28 * 28
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],  # [3, 4, 6, 3]中的6
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding  # layer 3降尺寸。因为stride =2要进行降采样。输出就是 1024 * 14 * 14
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],  # [3, 4, 6, 3]中的3
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,  # layer 4降尺寸。因为stride =2要进行降采样。输出就是 2048 * 7 * 7
            downsample_padding=downsample_padding
        )
        # 平均池化，核的大小为7x7
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        # 决定是否下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        # if self.dropout is not None:
        #     x = self.dropout(x)
        # 2048行，-1表示不确定为多少列
        x = x.view(x.size(0), -1)
        # num_classes
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


# 初始化预训练模型
def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], 'num_classes should be {}, but is {}'.format(settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


# se_resnet50
def se_resnet50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,inplanes=64,
                  downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)

    return model
