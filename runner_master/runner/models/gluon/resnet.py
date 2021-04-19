__all__ = ['ResNet']


import re
import torch.nn as nn
from functools import partial
from ...operators import get_gapool_op, get_flatten_op


# 从bag_of_tricks论文划分，3类skip connection：
#     1. ideneity
#     2. 原始ResNet的conv downsample
#     3. Bag of Tricks的avgpool downsample
# 从pre-act与否划分，2类skip connection：
#     1. 原始ResNet的post-activation模式，weight-bn-relu，最后的relu跟外面的block共享，略去，只剩weight-bn
#     2. pre-activation模式，bn-relu-conv，最前面的bn-relu跟外面的block共享，略去，只剩weight
def skip_connection(in_channels, out_channels, stride, avg_down, pre_act):
    if in_channels == out_channels and stride == 1:
        return nn.Sequential()
    elif stride == 1 or not avg_down:
        if not pre_act:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            )
    else:
        if not pre_act:
            return nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            return nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            )


class SqueezeExcitation(nn.Module):

    def __init__(self, channels, ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = get_gapool_op()
        neck_channels = max(1, channels // ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, neck_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.avgpool(x)
        w = self.mlp(w)
        return w * x


class BasicBlockV1(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shrink=None, num_groups=1, avg_down=False, neck_down=None):
        super(BasicBlockV1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, groups=num_groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.skip = skip_connection(in_channels, out_channels, stride, avg_down, pre_act=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.skip(x)
        out = self.relu2(out)
        return out


class BasicBlockV2(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shrink=None, num_groups=1, avg_down=False, neck_down=None):
        super(BasicBlockV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, groups=num_groups)
        self.skip = skip_connection(in_channels, out_channels, stride, avg_down, pre_act=True)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += self.skip(x)
        return out


class BottleneckV1(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shrink=4, num_groups=1, avg_down=False, neck_down=False):
        super(BottleneckV1, self).__init__()
        neck_channels = round(out_channels / shrink)
        if not neck_down:
            in_stride, neck_stride = stride, 1
        else:
            in_stride, neck_stride = 1, stride

        self.conv1 = nn.Conv2d(in_channels, neck_channels, 1, in_stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(neck_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(neck_channels, neck_channels, 3, neck_stride, 1, bias=False, groups=num_groups)
        self.bn2 = nn.BatchNorm2d(neck_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(neck_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.skip = skip_connection(in_channels, out_channels, stride, avg_down, pre_act=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.skip(x)
        out = self.relu3(out)
        return out


class BottleneckV2(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shrink=4, num_groups=1, avg_down=False, neck_down=False):
        super(BottleneckV2, self).__init__()
        neck_channels = round(out_channels / shrink)
        if not neck_down:
            in_stride, neck_stride = stride, 1
        else:
            in_stride, neck_stride = 1, stride

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, neck_channels, 1, in_stride, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(neck_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(neck_channels, neck_channels, 3, neck_stride, 1, bias=False, groups=num_groups)
        self.bn3 = nn.BatchNorm2d(neck_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(neck_channels, out_channels, 1, bias=False)
        self.skip = skip_connection(in_channels, out_channels, stride, avg_down, pre_act=True)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        out += self.skip(x)
        return out


def init_weight(module):
    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')


def input_stem(in_channels, out_channels, deep_stem, narrow_stem):
    stem = []
    if (deep_stem, narrow_stem) == (False, False):
        stem.append(nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False))
    elif (deep_stem, narrow_stem) == (False, True):
        stem.append(nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False))
    elif (deep_stem, narrow_stem) == (True, True):
        neck_channels = out_channels // 2
        stem.append(nn.Conv2d(in_channels, neck_channels, 3, 2, 1, bias=False))
        stem.append(nn.BatchNorm2d(neck_channels))
        stem.append(nn.ReLU(inplace=True))
        stem.append(nn.Conv2d(neck_channels, neck_channels, 3, 1, 1, bias=False))
        stem.append(nn.BatchNorm2d(neck_channels))
        stem.append(nn.ReLU(inplace=True))
        stem.append(nn.Conv2d(neck_channels, out_channels, 3, 1, 1, bias=False))
    else:
        raise ValueError('deep and wide stem is not supported')
    stem.append(nn.BatchNorm2d(out_channels))
    stem.append(nn.ReLU(inplace=True))
    stem.append(nn.MaxPool2d(3, 2, 0, ceil_mode=True))
    return nn.Sequential(*stem)


def residual_stem(block, in_channels, out_channels, layers, stride):
    if layers == 0:
        return nn.Sequential()
    seq = []
    seq.append(block(in_channels, out_channels, stride))
    for i in range(1, layers):
        seq.append(block(out_channels, out_channels, 1))
    return nn.Sequential(*seq)


class ResNet(nn.Module):

    def __init__(self, layers, base_channels, bottleneck,
                 shrink=4, num_groups=1,
                 deep_stem=False, narrow_stem=False, pre_act=False,
                 use_se=False, avg_down=False, neck_down=False, zero_gamma=False,
                 num_classes=1000):
        super(ResNet, self).__init__()
        if len(layers) < 2:
            raise ValueError('len(layers) = {} < 2'.format(len(layers)))
        self.num_layers = len(layers)

        if not bottleneck:
            mul = [1, 1]
        else:
            mul = [1, 4]
        for _ in range(len(layers) - 1):
            mul.append(mul[-1] * 2)
        channels = [m * base_channels for m in mul]

        self.input_stem = input_stem(3, channels[0], deep_stem, narrow_stem)
        Block = {
            (False, False): BasicBlockV1,
            (False, True): BasicBlockV2,
            (True, False): BottleneckV1,
            (True, True): BottleneckV2,
        }[(bottleneck, pre_act)]
        block = partial(Block, shrink=shrink, num_groups=num_groups, avg_down=avg_down, neck_down=neck_down)
        self.layer1 = residual_stem(block, channels[0], channels[1], layers[0], stride=1)
        for i in range(len(layers) - 1):
            lay = layers[i + 1]
            in_ch = channels[i + 1]
            out_ch = channels[i + 2]
            self.add_module('se{}'.format(i + 1), SqueezeExcitation(in_ch) if use_se else nn.Sequential())
            self.add_module('layer{}'.format(i + 2), residual_stem(block, in_ch, out_ch, lay, stride=2))

        if not pre_act:
            self.output_stem = nn.Sequential()
        else:
            self.output_stem = nn.Sequential(
                nn.BatchNorm2d(channels[4]),
                nn.ReLU(inplace=True),
            )
        self.avgpool = get_gapool_op(7)
        self.view = get_flatten_op()
        self.fc = nn.Linear(channels[-1], num_classes)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weight(m)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_gamma:
            for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                for j, block in enumerate(layer):
                    if type(block) in (BasicBlockV1, BasicBlockV2):
                        block.bn2.weight.data.zero_()
                    elif type(block) in (BottleneckV1, BottleneckV2):
                        block.bn3.weight.data.zero_()
                    elif type(block) == SqueezeExcitation:
                        pass
                    else:
                        raise RuntimeError('layer {} block {} has unknown type for zero_gamma: {}'.format(
                            i, j, type(block)))

    def forward(self, x, forward_mode='cls'):
        if forward_mode == 'cls':
            return self.forward_cls(x)
        if forward_mode == 'det':
            return self.forward_det(x)
        raise ValueError('unknown forward_mode: {}'.format(forward_mode))

    def forward_cls(self, x):
        x = self.input_stem(x)
        x = self.layer1(x)
        for i in range(self.num_layers - 1):
            x = getattr(self, 'se{}'.format(i + 1))(x)
            x = getattr(self, 'layer{}'.format(i + 2))(x)
        x = self.output_stem(x)
        x = self.avgpool(x)
        x = self.view(x)
        x = self.fc(x)
        return x

    def forward_det(self, x):
        x = self.input_stem(x)
        feature_maps = [x]
        x = self.layer1(x)
        for i in range(self.num_layers - 1):
            x = getattr(self, 'se{}'.format(i + 1))(x)
            x = getattr(self, 'layer{}'.format(i + 2))(x)
            feature_maps.append(x)
        return tuple(feature_maps)

    def get_cls_head(self):
        return self.fc

    def only_keep_cls_backbone(self):
        self.fc = nn.Sequential()
        return self

    def only_keep_cnn_backbone(self):
        self.avgpool = nn.Sequential()
        self.view = nn.Sequential()
        self.fc = nn.Sequential()
        return self

    def change_output_classes(self, num_classes, keep_if_same=True):
        if not isinstance(self.fc, nn.Linear):
            raise NotImplementedError('Now self.fc is not nn.Linear.')
        if num_classes != self.fc.out_features or not keep_if_same:
            fc = nn.Linear(self.fc.in_features, num_classes)
            init_weight(fc)
            if self.fc.weight.is_cuda:
                fc.cuda(self.fc.weight.device)
                assert fc.weight.device == self.fc.weight.device
            self.fc = fc
        return self

    def change_input_channels(self, in_channels):
        old_conv = self.input_stem[0]
        if in_channels != old_conv.in_channels:
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                old_conv.kernel_size, old_conv.stride, old_conv.padding,
                old_conv.dilation,
                old_conv.groups,
                old_conv.bias is not None,
                old_conv.padding_mode,
            )
            init_weight(new_conv)
            if old_conv.weight.is_cuda:
                new_conv.cuda()
            self.input_stem[0] = new_conv

    def load_state_dict(self, state_dict, *args, **kwargs):
        # load the parameters of old-version SE
        # where the `mlp` contains nn.Linear rather than nn.Conv2d
        pattern = 'se\d\.mlp\.\d\.weight'  # noqa
        for key, value in list(state_dict.items()):
            if re.fullmatch(pattern, key) and value.dim() == 2:
                state_dict[key] = value.unsqueeze(-1).unsqueeze(-1)
        super(ResNet, self).load_state_dict(state_dict, *args, **kwargs)


def __resnet__(model_name, layers, base_channels, bottleneck,
               shrink, num_groups,
               deep_stem, narrow_stem, pre_act,
               use_se, avg_down, neck_down):
    def f(pretrained=False, zero_gamma=False, num_classes=1000):
        model = ResNet(layers, base_channels, bottleneck,
                       shrink, num_groups,
                       deep_stem, narrow_stem, pre_act,
                       use_se, avg_down, neck_down, zero_gamma,
                       num_classes)
        if pretrained:
            import pdb; pdb.set_trace()
        return model

    f.__doc__ = \
        """Constructs a {} model.

        Structure Arguments:
            layers: {}
            base_channels: {}
            bottleneck: {}
            shrink: {}
            num_groups: {}
            deep_stem: {}
            narrow_stem: {}
            pre_act: {}
            use_se: {}
            avg_down: {}
            neck_down: {}

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            zero_gamma (bool): If True, zero-initialize the weight of last bn in each residual block
            num_classes (int): Number of output classes
        """.format(model_name, layers, base_channels, bottleneck,
                   shrink, num_groups,
                   deep_stem, narrow_stem, pre_act,
                   use_se, avg_down, neck_down)

    return f


se_configs = [
    ['', {'use_se': False}],
    ['se_', {'use_se': True}],
]
layer_configs = [
    ['18', {'layers': [2, 2, 2, 2], 'bottleneck': False}],
    ['34', {'layers': [3, 4, 6, 3], 'bottleneck': False}],
    ['50', {'layers': [3, 4, 6, 3], 'bottleneck': True}],
    ['101', {'layers': [3, 4, 23, 3], 'bottleneck': True}],
    ['152', {'layers': [3, 8, 36, 3], 'bottleneck': True}],
]
version_configs = [
    ['v1', {'pre_act': False}],
    ['v2', {'pre_act': True}],
]
tweak_configs = [
    ['a', {'deep_stem': False, 'narrow_stem': False, 'avg_down': False, 'neck_down': False}],
    ['b', {'deep_stem': False, 'narrow_stem': False, 'avg_down': False, 'neck_down': True}],
    ['c', {'deep_stem': True, 'narrow_stem': True, 'avg_down': False, 'neck_down': True}],
    ['d', {'deep_stem': True, 'narrow_stem': True, 'avg_down': True, 'neck_down': True}],
]
width_configs = [
    ['_4by1', {'base_channels': 256}],
    ['_2by1', {'base_channels': 128}],
    ['', {'base_channels': 64}],
    ['_1by2', {'base_channels': 32}],
    ['_3by8', {'base_channels': 24}],
    ['_1by4', {'base_channels': 16}],
    ['_1by8', {'base_channels': 8}],
    ['_1by16', {'base_channels': 4}],
]
group_configs = [
    ['', {'num_groups': 1}],
    ['_g2', {'num_groups': 2}],
    ['_g4', {'num_groups': 4}],
    ['_g8', {'num_groups': 8}],
    ['_g16', {'num_groups': 16}],
    ['_g32', {'num_groups': 32}],
    ['_g64', {'num_groups': 64}],
]
shrink_configs_1 = [
    ['', {'shrink': 1}],
]
shrink_configs_2 = [
    ['_s1', {'shrink': 1}],
    ['_s2', {'shrink': 2}],
    ['', {'shrink': 4}],
    ['_s8', {'shrink': 8}],
]
shrink_configs_dict = {
    '18': shrink_configs_1,
    '34': shrink_configs_1,
    '50': shrink_configs_2,
    '101': shrink_configs_2,
    '152': shrink_configs_2,
}
local = locals()
for SE, se in se_configs:
    for LAY, lay in layer_configs:
        for VER, ver in version_configs:
            for TW, tw in tweak_configs:
                for WID, wid in width_configs:
                    for GRP, grp in group_configs:
                        for SHR, shr in shrink_configs_dict[LAY]:
                            if LAY in ['18', '34']:
                                min_neck_channels = round(wid['base_channels'] / shr['shrink'])
                            else:
                                min_neck_channels = round(wid['base_channels'] * 4 / shr['shrink'])
                            if min_neck_channels % grp['num_groups'] != 0:
                                continue
                            model_name = '{SE}resnet{LAY}_{VER}{TW}{WID}{SHR}{GRP}'.format(
                                SE=SE, LAY=LAY, VER=VER, TW=TW, WID=WID, SHR=SHR, GRP=GRP,
                            )
                            assert model_name not in local
                            local[model_name] = __resnet__(
                                'gluon.' + model_name,
                                **lay, **wid, **ver, **tw, **se, **shr, **grp,
                            )
                            __all__.append(model_name)


std_nick_names = []
std_nick_names.extend([
    ('resnet18_v1b', 'resnet18'),
    ('resnet34_v1b', 'resnet34'),
    ('resnet50_v1b', 'resnet50'),
    ('resnet101_v1b', 'resnet101'),
    ('resnet152_v1b', 'resnet152'),
])
for lay in ('50', '101', '152'):
    std_nick_names.extend([
        ('resnet{}_v1b'.format(lay), 'resnext{}_1x64d'.format(lay)),
        ('resnet{}_v1b_g2'.format(lay), 'resnext{}_2x32d'.format(lay)),
        ('resnet{}_v1b_g4'.format(lay), 'resnext{}_4x16d'.format(lay)),
        ('resnet{}_v1b_g8'.format(lay), 'resnext{}_8x8d'.format(lay)),
        ('resnet{}_v1b_g16'.format(lay), 'resnext{}_16x4d'.format(lay)),
        ('resnet{}_v1b_g32'.format(lay), 'resnext{}_32x2d'.format(lay)),
        ('resnet{}_v1b_g64'.format(lay), 'resnext{}_64x1d'.format(lay)),

        ('resnet{}_v1b_s2'.format(lay), 'resnext{}_1x128d'.format(lay)),
        ('resnet{}_v1b_s2_g2'.format(lay), 'resnext{}_2x64d'.format(lay)),
        ('resnet{}_v1b_s2_g4'.format(lay), 'resnext{}_4x32d'.format(lay)),
        ('resnet{}_v1b_s2_g8'.format(lay), 'resnext{}_8x16d'.format(lay)),
        ('resnet{}_v1b_s2_g16'.format(lay), 'resnext{}_16x8d'.format(lay)),
        ('resnet{}_v1b_s2_g32'.format(lay), 'resnext{}_32x4d'.format(lay)),
        ('resnet{}_v1b_s2_g64'.format(lay), 'resnext{}_64x2d'.format(lay)),

        ('resnet{}_v1b_s1'.format(lay), 'resnext{}_1x256d'.format(lay)),
        ('resnet{}_v1b_s1_g2'.format(lay), 'resnext{}_2x128d'.format(lay)),
        ('resnet{}_v1b_s1_g4'.format(lay), 'resnext{}_4x64d'.format(lay)),
        ('resnet{}_v1b_s1_g8'.format(lay), 'resnext{}_8x32d'.format(lay)),
        ('resnet{}_v1b_s1_g16'.format(lay), 'resnext{}_16x16d'.format(lay)),
        ('resnet{}_v1b_s1_g32'.format(lay), 'resnext{}_32x8d'.format(lay)),
        ('resnet{}_v1b_s1_g64'.format(lay), 'resnext{}_64x4d'.format(lay)),
    ])
std_nick_names.extend([
    ('resnet18_v1b_2by1', 'wide_resnet18_x2'),
    ('resnet18_v1b_4by1', 'wide_resnet18_x4'),
    ('resnet34_v1b_2by1', 'wide_resnet34_x2'),
    ('resnet34_v1b_4by1', 'wide_resnet34_x4'),
    ('resnet50_v1b_s2', 'wide_resnet50_x2'),
    ('resnet50_v1b_s1', 'wide_resnet50_x4'),
    ('resnet101_v1b_s2', 'wide_resnet101_x2'),
    ('resnet101_v1b_s1', 'wide_resnet101_x4'),
    ('resnet152_v1b_s2', 'wide_resnet152_x2'),
    ('resnet152_v1b_s1', 'wide_resnet152_x4'),
])
for std_name, nick_name in std_nick_names:
    assert nick_name not in local
    local[nick_name] = local[std_name]
    __all__.append(nick_name)
