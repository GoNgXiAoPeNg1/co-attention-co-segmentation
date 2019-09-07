import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import pdb
from utils.resnet import resnet50

resnet_channel = 512


# 使用了vgg16 和resnet50   vgg16训练较快
# 修改反卷积cat
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # encoder

        self.pretrained_model = vgg16(pretrained=True)
        self.features  = list(self.pretrained_model.features.children(
        ))
        self.features_map = nn.Sequential(*self.features)

        # self.features_map = resnet50(pretrained=True)


        # self attention generation
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_avg_pool = nn.Conv2d(resnet_channel, 1, kernel_size=3, padding=1)
        self.mlp1 = nn.Linear(resnet_channel, 4096)
        self.mlp2 = nn.Linear(4096, resnet_channel)
        self.upsample = nn.Upsample(16)

        # decoder
        self.dec = Decoder(2, resnet_channel, 2, activ='relu', pad_type='reflect')

    def forward(self, x,y):

        # reconstruct an image
        vgg_x, vgg_y, vgg_x_weight, vgg_y_weight = self.encode(x, y)
        # spatial information
        spatial_avg_pool_x = self.spatial_avg_pool(vgg_x)
        spatial_avg_pool_y = self.spatial_avg_pool(vgg_y)

        images_recon_x, images_recon_y = self.decode(
            vgg_x, vgg_y, vgg_x_weight, vgg_y_weight, spatial_avg_pool_x, spatial_avg_pool_y)
        return images_recon_x, images_recon_y

    def generate_attention(self, x):
        vgg_x = self.features_map(x)

        feature_x = self.global_avg_pool(vgg_x)
        #去掉上采样  用广播 之后可以去掉mlp，可以增强attention的作用力
        attention_x = F.sigmoid(self.mlp2(
            F.tanh(self.mlp1(feature_x.view(-1, resnet_channel))))).view(-1, resnet_channel, 1, 1)

        # attention_x = self.upsample(F.sigmoid(self.mlp2(
        #     F.tanh(self.mlp1(feature_x.view(-1, 512))))).view(-1, 512, 1, 1))
        return vgg_x,attention_x

    def encode(self, x, y):
        # encode an image to its content
        vgg_x,attention_x = self.generate_attention(x)
        vgg_y,attention_y = self.generate_attention(y)
        return vgg_x, vgg_y, attention_x, attention_y


    def decode(self, vgg_x, vgg_y, attention_x, attention_y, spatial_avg_pool_x, spatial_avg_pool_y):

        # mask_x = self.dec(attention_y * vgg_x)
        # mask_y = self.dec(attention_x * vgg_y)

        mask_x = self.dec(self.generate_mask_multi_num(attention_y, vgg_x, multi_num=5))
        mask_y = self.dec(self.generate_mask_multi_num(attention_x, vgg_y, multi_num=5))

        #print("image size:",images.size())
        return mask_x, mask_y

    #我的实现

    def generate_attention_gaussion(self, attention_x, attentin_y):
        '''
        加入高斯距离后，取指数函数  生成的attention只保留同时相似的物体
        :param attention_x:
        :param attentin_y:
        :return: attention
        '''
        attention = torch.exp(-torch.pow((attention_x-attentin_y),2))
        return attention
    def generate_mask1(self, vgg_x, vgg_y, channel_num_eachblock=1):
        '''
        输入两个encoder的结果，返回vgg_y，vgg_x作为kernal
        :param vgg_x:
        :param vgg_y:
        :return:
        '''
        block_num = 512 / channel_num_eachblock
        for i in block_num:
            # vgg_y = vgg_x[i*channel_num_eachblock:channel_num_eachblock]*vgg_y
            vgg_y = F.relu(vgg_x[i*channel_num_eachblock:channel_num_eachblock]*vgg_y)
        return vgg_y

    def generate_mask_multi_num(self, attention_x, vgg_y, multi_num=1):
        '''
        :param attention_x:
        :param vgg_y:
        :param multi_num: 通过多次与vgg_y相乘，可去掉不相似的物体
        :return:
        '''

        for _ in range(multi_num):# relu_num 个数可以增强attention，相对降低spatial
            # vgg_y = attention_x*vgg_y
            vgg_y = F.relu(attention_x*vgg_y)  #F.leaklyRelu

        return vgg_y

    def generate_mask_shift(self, attention_x, vgg_y, shift_num=0):
        '''
        #
        :param attention_x:
        :param vgg_y:
        :param shift_num: attention—x 上下移动，可保证attention每个点一定范围内都可能是狗
        :return:
        '''
        pdb.set_trace()

        for i in range(1,shift_num):
            temp = attention_x[:,0:-i,:,:]
            temp = torch.cat((temp,torch.ones(1, i)),)
            # vgg_y = temp * vgg_y
            vgg_y = F.relu(temp * vgg_y)
        for i in range(shift_num):
            temp = attention_x[i:]
            temp = temp.insert(0, torch.ones(1, i))
            vgg_y = temp * vgg_y
            vgg_y = F.relu(temp * vgg_y)
        return vgg_y

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='bn', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm,
                                    activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_res, dim, output_dim, activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, 'bn', activ, pad_type=pad_type)]

        for i in range(5):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='bn', activation=activ, pad_type='reflect')]
            dim //= 2

        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3,
                                   norm='none', activation='none', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='bn', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm,
                              activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm,
                              activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)

        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim,
                              kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
