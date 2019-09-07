import pdb
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import torch
from torch.functional import F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

'''
post process  super resolution
'''



def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


'''
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
segmentation part
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
'''


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int, activate=True):
        super(ConvRelu, self).__init__()
        self.activate = activate
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activate:
            x = self.bn(x)
            x = self.activation(x)
        return x


class DecoderBlockResnet(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlockResnet, self).__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels, activate=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UnetResNet(nn.Module):
    '''
    the size of input must be 32  +
    '''

    def __init__(self, num_classes=2, num_filters=32, pretrained=True, Dropout=.2, model="resnet50"):
        super(UnetResNet, self).__init__()

        self.encoder = torchvision.models.resnet50(pretrained=pretrained)

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        # self.center = DecoderBlockResnet(2048, num_filters * 8 * 2,num_filters * 8)
        self.center = ConvRelu(2048, num_filters * 8, activate=True)

        # self attention generation

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp1 = nn.Linear(512, 4096)
        self.mlp2 = nn.Linear(4096, 256)
        self.upsample = nn.Upsample(16)

        self.dec5 = DecoderBlockResnet(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlockResnet(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlockResnet(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlockResnet(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2)
        self.dec1 = DecoderBlockResnet(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose2d):
                # pdb.set_trace()
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, z=None):
        conv1_1 = self.conv1(x1)
        conv1_2 = self.conv1(x2)
        conv2_1 = self.conv2(conv1_1)
        conv2_2 = self.conv2(conv1_2)
        conv3_1 = self.conv3(conv2_1)
        conv3_2 = self.conv3(conv2_2)
        conv4_1 = self.conv4(conv3_1)
        conv4_2 = self.conv4(conv3_2)
        conv5_1 = self.conv5(conv4_1)
        conv5_2 = self.conv5(conv4_2)

        # center_1 = self.center(self.pool(conv5_1))
        # center_2 = self.center(self.pool(conv5_2))
        center_1 = self.center(conv5_1)
        center_2 = self.center(conv5_2)

        feature_1 = self.global_avg_pool(center_1)
        feature_2 = self.global_avg_pool(center_2)
        feature_12 = torch.cat((feature_1, feature_2), dim=1)
        # pdb.set_trace()
        attention_12 = F.sigmoid(self.mlp2(
            F.tanh(self.mlp1(feature_12.view(-1, 512))))).view(-1, 256, 1, 1)

        center_1 = center_1 * attention_12
        center_2 = center_2 * attention_12

        dec5_1 = self.dec5(torch.cat([center_1, conv5_1], 1))
        dec5_2 = self.dec5(torch.cat([center_2, conv5_2], 1))
        dec4_1 = self.dec4(torch.cat([dec5_1, conv4_1], 1))
        dec4_2 = self.dec4(torch.cat([dec5_2, conv4_2], 1))
        dec3_1 = self.dec3(torch.cat([dec4_1, conv3_1], 1))
        dec3_2 = self.dec3(torch.cat([dec4_2, conv3_2], 1))
        dec2_1 = self.dec2(torch.cat([dec3_1, conv2_1], 1))
        dec2_2 = self.dec2(torch.cat([dec3_2, conv2_2], 1))

        dec1_1 = self.dec1(dec2_1)
        dec1_2 = self.dec1(dec2_2)
        dec0_1 = self.dec0(dec1_1)
        dec0_2 = self.dec0(dec1_2)

        return self.final(dec0_1), self.final(dec0_2)

    # 我的实现

    def generate_attention_gaussion(self, attention_x, attentin_y):
        '''
        加入高斯距离后，取指数函数  生成的attention只保留同时相似的物体
        :param attention_x:
        :param attentin_y:
        :return: attention
        '''
        # sigma越小，x与y差距增大时，值减小的更多
        sigma = 2
        attention = torch.exp((-torch.pow((attention_x - attentin_y), 2))/(2*sigma**2))/sigma
        attention_x = attention * attention_x
        attention_y = attention * attention_y
        return attention

    def generate_mask_shift(self, attention_x, vgg_y, shift_num=0):
        '''
        #
        :param attention_x:
        :param vgg_y:
        :param shift_num: attention—x 上下移动，可保证attention每个点一定范围内都可能是狗
        :return:
        '''

        for i in range(1, shift_num):
            temp = attention_x[:, 0:-i, :, :]
            temp = torch.cat((temp, torch.ones(1, i)), )
            # vgg_y = temp * vgg_y
            vgg_y = F.relu(temp * vgg_y)
        for i in range(shift_num):
            temp = attention_x[i:]
            temp = temp.insert(0, torch.ones(1, i))
            vgg_y = temp * vgg_y
            vgg_y = F.relu(temp * vgg_y)
        return vgg_y