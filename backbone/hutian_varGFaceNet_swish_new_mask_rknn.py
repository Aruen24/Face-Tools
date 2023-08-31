# coding:utf-8
# 参考： https://www.cnblogs.com/wanghui-garcia/p/12582953.html 人脸检测和识别以及检测中loss学习 - 14 - VarGFaceNet: An Efficient Variable Group Convolutional Neural Network for Lightweight Face Recognition - 2 - 代码
#        https://blog.csdn.net/jacke121/article/details/102897992  VarGFaceNet
import torch.nn as nn
import torch

class SwishV1(nn.Module):

    def __init__(self):
        super(SwishV1, self).__init__()

    def forward(self, feat):
        return feat * torch.sigmoid(feat)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class se_block(nn.Module):
    def __init__(self, channels, reduction):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = SwishV1() # changed

        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

# Head setting的上面分支
class separable_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, expansion=1, stride=1, dw_bn_out=True,
                 dw_relu_out=True, pw_bn_out=True, pw_relu_out=True, group_base=1): # group_base changed to 1 as hutian
        super(separable_conv2d, self).__init__()
        # depthwise
        assert in_channels % group_base == 0
        self.dw_conv = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False, groups=in_channels // group_base)
        if dw_bn_out:
            self.dw_bn = nn.BatchNorm2d(num_features = in_channels * expansion, eps = 2e-5, momentum = 0.9, affine = True) # changed as hutian
        else:
            self.dw_bn = nn.Sequential()
        if dw_relu_out:
            self.dw_relu = SwishV1() # old version same , changed
        else:
            self.dw_relu = nn.Sequential()

        # pointwise
        self.pw_conv = nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        if pw_bn_out:
            self.pw_bn = nn.BatchNorm2d(num_features = out_channels, eps = 2e-5, momentum = 0.9, affine = True) # changed as hutian
        else:
            self.pw_bn = nn.Sequential()
        if pw_relu_out:
            self.pw_relu = SwishV1() # changed
        else:
            self.pw_relu = nn.Sequential()

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_relu(x)

        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)

        return x

# Norm Block
class vargnet_block(nn.Module):
    def __init__(self, channels_1, channels_2, channels_3, reduction, expansion=2, multiplier=1, kernel_size=3,
                 stride=1, dilate=1, dim_match=True, use_se=True):
        super(vargnet_block, self).__init__()
        pad = ((kernel_size - 1) * dilate + 1) // 2
        if not dim_match:
            self.short_cut = separable_conv2d(int(channels_1 * multiplier), int(channels_3 * multiplier),
                                              kernel_size=kernel_size, padding=pad, expansion=expansion, stride=stride,
                                              pw_relu_out=False, group_base=1)
        else:
            self.short_cut = nn.Sequential()
        self.part_1 = separable_conv2d(int(channels_1 * multiplier), int(channels_2 * multiplier),
                                       kernel_size=kernel_size, padding=pad, expansion=expansion, stride=stride, group_base=1)
        self.part_2 = separable_conv2d(int(channels_2 * multiplier), int(channels_3 * multiplier),
                                       kernel_size=kernel_size, padding=pad, expansion=expansion, stride=1,
                                       pw_relu_out=False, group_base=1)
        if use_se:
            self.se = se_block(int(channels_3 * multiplier), reduction)
        else:
            self.se = nn.Sequential()
        self.relu = SwishV1() # changed

    def forward(self, x):
        short_cut_data = self.short_cut(x)
        x = self.part_1(x)
        x = self.part_2(x)
        x = self.se(x)
        x = self.relu(short_cut_data + x)

        return x


# Down sampling block
class vargnet_branch_merge_block(nn.Module):
    def __init__(self, channels_1, channels_2, channels_3, expansion=2, multiplier=1, kernel_size=3, stride=2, dilate=1,
                 dim_match=False):
        super(vargnet_branch_merge_block, self).__init__()
        pad = ((kernel_size - 1) * dilate + 1) // 2
        if not dim_match:
            self.short_cut = separable_conv2d(int(channels_1 * multiplier), int(channels_3 * multiplier),
                                              kernel_size=kernel_size, padding=pad, expansion=expansion, stride=stride,
                                              pw_relu_out=False, group_base=1)
        else:
            self.short_cut = nn.Sequential()
        self.part_1_branch_1 = separable_conv2d(int(channels_1 * multiplier), int(channels_2 * multiplier),
                                                kernel_size=kernel_size, padding=pad, expansion=expansion,
                                                stride=stride, pw_relu_out=False, group_base=1)
        # self.part_1_branch_2 = separable_conv2d(int(channels_1 * multiplier), int(channels_2 * multiplier),
                                                # kernel_size=kernel_size, padding=pad, expansion=expansion,
                                                # stride=stride, pw_relu_out=False) # 胡天把这个down-sample中的这个分支去掉了
        self.relu_1 = SwishV1() # changed

        self.part_2 = separable_conv2d(int(channels_2 * multiplier), int(channels_3 * multiplier),
                                       kernel_size=kernel_size, padding=pad, expansion=expansion, stride=1,
                                       pw_relu_out=False, group_base=1)
        self.relu_2 = SwishV1() # changed

    def forward(self, x):
        short_cut_data = self.short_cut(x)
        x_branch_1 = self.part_1_branch_1(x)
        # x_branch_2 = self.part_1_branch_2(x) # 因为胡天把分支2这个去掉了，所以要注释掉
        # x = self.relu_1(x_branch_1 + x_branch_2) # 因为胡天把分支2这个去掉了，所以要注释掉
        x = self.relu_1(x_branch_1)
        x = self.part_2(x)
        x = self.relu_2(short_cut_data + x)
        return x

# Down sampling block(1个) + Norm Block(n个)
class add_vargnet_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, norm_block_number, reduction, expansion=2, multiplier=1,
                 kernel_size=3, stride=2, dilate=1):
        super(add_vargnet_conv_block, self).__init__()
        self.down_sample_block = vargnet_branch_merge_block(in_channels, out_channels, out_channels,
                                                            expansion=expansion, multiplier=multiplier,
                                                            kernel_size=kernel_size, stride=stride, dilate=dilate,
                                                            dim_match=False)

        norm_blocks = []
        for i in range(norm_block_number - 1):
            norm_blocks.append(vargnet_block(out_channels, out_channels, out_channels, reduction, expansion=expansion,
                                             multiplier=multiplier, kernel_size=kernel_size, stride=1, dilate=dilate,
                                             dim_match=True, use_se=True))
        self.norm_blocks_layer = nn.Sequential(*norm_blocks)

    def forward(self, x):
        x = self.down_sample_block(x)
        x = self.norm_blocks_layer(x)
        return x

# Head_setting
class add_head_block(nn.Module):
    def __init__(self, channels, multiplier, reduction, kernel_size=3, stride=1, padding=1):
        super(add_head_block, self).__init__()
        self.conv1 = nn.Conv2d(3, int(channels * multiplier), kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features = int(channels * multiplier), eps = 2e-5, momentum = 0.9, affine = True) # changed as hutian
        self.relu1 = SwishV1() # old_version: ReLU6 changed

        self.head = vargnet_block(channels, channels, channels, reduction, expansion=1, multiplier=multiplier,
                                  kernel_size=kernel_size, stride=2, dim_match=False, use_se=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.head(x)

        return x

# embedding setting
class add_emb_block(nn.Module):
    def __init__(self, in_channels, last_channels, emb_size, group_base=1): # changed group_base = 8 as hutian
        super(add_emb_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, last_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features = last_channels, eps = 2e-5, momentum = 0.9, affine = True) # changed as hutian
        self.relu1 = SwishV1() # changed

        # depthwise
        # self.dw_conv_mask = nn.Conv2d(last_channels, last_channels, kernel_size=(5,9), stride=1, padding=0, bias=False,
        #                          groups=last_channels // group_base)

        # gap 代替 depthwise
        self.gap_mask_rknn = nn.AdaptiveAvgPool2d((1, 1))

        self.dw_bn = nn.BatchNorm2d(last_channels, eps = 2e-5, momentum = 0.9, affine = True) # changed as hutian

        # pointwise
        # self.pw_conv = nn.Conv2d(last_channels, last_channels // 2, kernel_size=1, stride=1, padding=0, bias=False) # 胡天没有做逐点卷积
        # self.pw_bn = nn.BatchNorm2d(last_channels // 2, eps = 2e-05, momentum = 0.9, affine = True) # changed as hutian 胡天没有做逐点卷积
        self.pw_relu = SwishV1() # changed

        # self.fc = nn.Linear(last_channels // 2, emb_size, bias=False)
        self.fc = nn.Linear(last_channels, emb_size, bias=False)
        self.bn = nn.BatchNorm1d(emb_size, eps = 2e-5, momentum = 0.9, affine = True) # changed as hutian

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # x = self.dw_conv_mask(x)
        x = self.gap_mask_rknn(x)
        x = self.dw_bn(x)

        # x = self.pw_conv(x)
        # x = self.pw_bn(x)
        x = self.pw_relu(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.bn(x)

        return x

class VarGFaceNet(nn.Module):
    def __init__(self, last_channels, emb_size, filter_list, norm_block_number, multiplier, reduction, num_stage,
                 expansion, if_softmax = False, if_l2_norm = False, classes = 1000):
        super(VarGFaceNet, self).__init__()
        self.head = add_head_block(filter_list[0], multiplier, reduction, kernel_size=3, stride=1, padding=1)
        
        body = []
        for i in range(num_stage):
            body.append(add_vargnet_conv_block(filter_list[i], filter_list[i + 1], norm_block_number[i], reduction,
                                               expansion=expansion, multiplier=multiplier, kernel_size=3, stride=2,
                                               dilate=1))
        self.body_layer = nn.Sequential(*body)

        self.embedding = add_emb_block(int(filter_list[num_stage] * multiplier), last_channels, emb_size, group_base=1) # changed group_base = 8 as hutian
        
        self.if_l2_norm = if_l2_norm
        self.if_softmax = if_softmax
        if self.if_softmax:
            self.classifer = nn.Linear(emb_size, classes)
            
        self._initialize_weights()

    def forward(self, x):
        x = self.head(x)
        x = self.body_layer(x)
        x = self.embedding(x)
        if self.if_softmax:
            x = self.classifer(x)
        if self.if_l2_norm:
            x = l2_norm(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # 如果使用cosface训练使用这个
                # nn.init.normal_(m.weight, 0, 0.1) # 如果使用ArcFace训练使用这个
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def varGFaceNet_swish_new(if_softmax = False, if_l2_norm = False, multiplier = 1.25, classes = 1000):
    """
        last_channels: 最后一层的通道数,默认1024
        emb_size: 嵌入特征的大小,默认512
        filter_list:每一层的通道数
        norm_block_number:每个num_stage中norm block的个数
        multiplier:乘于filter_list中的通道数，起到拓宽网络的作用,可0.5、1、1.5等
        reduction:SE模块中通道的缩小倍数, 默认4
        num_stage:中间的Down sampling block(1个) + Norm Block(n个)的个数,默认3
        expansion:下一层的通道数 = 上一层的通道数*expansion,默认2倍
    """
    filter_list = [32, 64, 128, 256]
    norm_block_number = [3, 7, 4]
    return VarGFaceNet(last_channels=1024, emb_size=512, filter_list=filter_list, norm_block_number=norm_block_number,
                       multiplier=multiplier, reduction=4, num_stage=3, expansion=2, if_softmax = if_softmax, if_l2_norm = if_l2_norm, classes = classes)

if __name__ == '__main__':
    model = varGFaceNet_swish_new()
    for name, child in model.named_children():
        print(name)
        print(child)
    input = torch.autograd.Variable(torch.randn(4, 3, 112, 112))
    output = model(input)
    print(output.shape)
