import torch
from torch import nn
from torchvision.models.utils import load_state_dict_from_url
# from model import l2_norm

__all__ = ['iresnet34', 'iresnet50', 'iresnet100']

model_urls = {
    'iresnet34': 'https://sota.nizhib.ai/insightface/iresnet34-5b0d0e90.pth',
    'iresnet50': 'https://sota.nizhib.ai/insightface/iresnet50-7f187506.pth',
    'iresnet100': 'https://sota.nizhib.ai/insightface/iresnet100-73e07ba7.pth'
}

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

# GDC module 
class GDC_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(GDC_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_features = out_c, eps = 2e-5, momentum = 0.9, affine = True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
        
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
                     
class IBasicBlock_first(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1):
        super(IBasicBlock_first, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            
        self.bn0= nn.BatchNorm2d( #
            num_features = inplanes,
            eps = 2e-5,
            momentum = 0.9, 
            affine = True
        )
        self.conv0 = conv1x1(inplanes, int(planes * 0.25))
        self.bn1 = nn.BatchNorm2d(
            num_features = int(planes * 0.25),
            eps = 2e-5,
            momentum = 0.9, 
            affine = True
        )
        self.conv1 = conv3x3(int(planes * 0.25), int(planes * 0.25))
        self.bn2 = nn.BatchNorm2d(
            num_features = int(planes * 0.25),
            eps = 2e-5,
            momentum = 0.9, 
            affine = True
        )
        self.prelu = nn.PReLU(int(planes * 0.25))
        self.conv2 = conv1x1(int(planes * 0.25), planes, stride)
        self.bn3 = nn.BatchNorm2d(
            num_features = planes,
            eps = 2e-5,
            momentum = 0.9, 
            affine = True
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.bn0(x)
        out = self.conv0(out)
        out = self.bn1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out
        
class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            
        self.bn0= nn.BatchNorm2d( #
            num_features = planes,
            eps = 2e-5,
            momentum = 0.9, 
            affine = True
        )
        self.conv0 = conv1x1(planes, int(planes*0.25))
        self.bn1 = nn.BatchNorm2d(
            num_features = int(planes*0.25),
            eps = 2e-5,
            momentum = 0.9, 
            affine = True
        )
        self.conv1 = conv3x3(int(planes*0.25), int(planes*0.25))
        self.bn2 = nn.BatchNorm2d(
            num_features = int(planes*0.25),
            eps = 2e-5,
            momentum = 0.9, 
            affine = True
        )
        self.prelu = nn.PReLU(int(planes*0.25))
        self.conv2 = conv1x1(int(planes*0.25), planes, stride)
        self.bn3 = nn.BatchNorm2d(
            num_features = planes,
            eps = 2e-5,
            momentum = 0.9, 
            affine = True
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.bn0(x)
        out = self.conv0(out)
        out = self.bn1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block,
                 layers,
                 if_softmax = False, if_l2_norm = False, classes=1000,
                 num_features=512,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None):
        super(IResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features = self.inplanes, eps = 2e-5, momentum = 0.9, affine = True)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       512,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       1024,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       2048,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.bn2 = nn.BatchNorm2d(
            num_features = 512 * block.expansion,
            eps = 2e-5,
            momentum = 0.9, 
            affine = True
        )
        self.dropout = nn.Dropout(p=0.4, inplace=True)
        self.GDC = GDC_block(in_c = 2048 * block.expansion, out_c = 512 * block.expansion, 
                            kernel = (7, 7), stride = (1, 1), padding = (0, 0), groups = 512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale,
                            num_features)
        self.fc_last = nn.Linear(512 * block.expansion, num_features)
        self.bn3 = nn.BatchNorm1d(
            num_features = num_features,
            eps = 2e-5,
            momentum = 0.9, 
            affine = False
        )
        
        self.if_l2_norm = if_l2_norm
        self.if_softmax = if_softmax
        if self.if_softmax:
            self.classifer = nn.Linear(512, classes)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight,
                                        # mode='fan_out',
                                        # nonlinearity='relu') # 如果使用cosface训练使用这个
                nn.init.normal_(m.weight, 0, 0.1) # 如果使用ArcFace训练使用这个
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, stage = 0, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(
                    num_features = planes * block.expansion,
                    eps = 2e-5,
                    momentum = 0.9, 
                    affine = True
                ),
            )
        
        layers = []
        layers.append(
            IBasicBlock_first(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      # planes * block.expansion * 0.25,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # N, 2048, 7, 7

        # x = self.bn2(x)
        # x = torch.flatten(x, 1) # 7 x 7 conv,  512 fc
        # x = self.dropout(x)
        x = self.GDC(x)
        x = x = x.view(x.shape[0], -1) # 
        # x = self.fc(x)
        x = self.fc_last(x)
        x = self.bn3(x)
        
        if self.if_softmax:
            x = self.classifer(x)
        if self.if_l2_norm:
            x = l2_norm(x)

        return x

def _iresnet(arch, block, layers, pretrained, if_softmax = False, if_l2_norm = False, classes = 100, progress=True, **kwargs):
    model = IResNet(block, layers, if_softmax = if_softmax, if_l2_norm = if_l2_norm, classes = classes, **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('resnet预训练模型已加载')
    return model

def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)

def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)

def iresnet100(pretrained=False, if_softmax = False, if_l2_norm = False, classes = 100, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained, if_softmax = if_softmax, if_l2_norm = if_l2_norm, classes = classes, 
                    progress=progress, **kwargs)

def iresnet269(pretrained=False, if_softmax = False, if_l2_norm = False, classes = 100, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 30, 48, 8], pretrained, if_softmax = if_softmax, if_l2_norm = if_l2_norm, classes = classes, 
                    progress=progress, **kwargs)
