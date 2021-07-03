import numpy
import torch
import torch.nn as nn

import functools
import torch.nn.functional as F
import spherical as S360

# adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py 

class ResNet360(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        depth=5,
        wf=32,
        conv_type='coord',
        padding='kernel',
        norm_type='none',
        activation='elu',
        up_mode='upconv',
        down_mode='downconv',
        width=512,
        use_dropout=False,
        padding_type='reflect',
        aspp=False
    ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(depth >= 0)
        super(ResNet360, self).__init__()
        model = (
            [
                create_conv(in_channels, wf, conv_type, \
                    kernel_size=7, padding=3, stride=1, width=width),
                create_normalization(wf, norm_type),
                create_activation(activation)
            ]
        )
        model += [ResnetBlock(wf, activation=activation, \
                norm_type=norm_type, conv_type=conv_type, \
                width=width)]
        n_downsampling = 2
        for i in range(n_downsampling): 
            mult = 2 ** i
            model += (
                [
                    create_conv(wf * mult, wf * mult * 2, conv_type, \
                        kernel_size=3, stride=2, padding=1, width=width // (i+1)),
                    create_normalization(wf * mult * 2, norm_type),
                    create_activation(activation)
                ]
            )
            model += [ResnetBlock(wf * mult * 2, activation=activation, \
                norm_type=norm_type, conv_type=conv_type, \
                width=width // (2 ** (i+1)))]
            model += [ResnetBlock(wf * mult * 2, activation=activation, \
                norm_type=norm_type, conv_type=conv_type, \
                width=width // (2 ** (i+1)))]
            #model += [ResnetBlock(wf * mult * 2, activation=activation, \
            #    norm_type=norm_type, conv_type=conv_type, \
            #    width=width // (2 ** (i+1)))]
        mult = 2 ** n_downsampling
        for i in range(depth):
            model += [ResnetBlock(wf * mult, activation=activation, \
                norm_type=norm_type, conv_type=conv_type, \
                width=width // (2 ** n_downsampling))]
        
        #aspp
        if aspp:
            model += [ASPP(wf*mult, wf*mult)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            '''
            model += (
                [
                    nn.ConvTranspose2d(wf * mult, int(wf * mult / 2),
                        kernel_size=3, stride=2,
                        padding=1, output_padding=1),
                    create_normalization(int(wf * mult / 2), norm_type),
                    create_activation(activation)
                ]
            )
            model += [ResnetBlock(int(wf * mult / 2), activation=activation, \
                norm_type=norm_type, conv_type=conv_type, \
                width=width // (mult // 2))]
            '''
            model += (
                [
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(wf * mult, int(wf * mult / 2),
                        kernel_size=3, stride=1,
                        padding=1),
                    create_normalization(int(wf * mult / 2), norm_type),
                    create_activation(activation),
                    nn.Conv2d(int(wf * mult / 2), int(wf * mult / 2),
                        kernel_size=3, stride=1,
                        padding=1),
                    create_normalization(int(wf * mult / 2), norm_type),
                    create_activation(activation)
                ]
            )
            

        model += [create_conv(wf, out_channels, conv_type, \
            kernel_size=7, padding=3, width=width)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        #print(self.model)
        #exit()
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, norm_type, conv_type, activation, width):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block +=(
            [
                create_conv(dim, dim, conv_type, width=width),
                create_normalization(dim, norm_type),
                create_activation(activation),
            ]
        )
        conv_block +=(
            [
                create_conv(dim, dim, conv_type, width=width),
                create_normalization(dim, norm_type),
            ]
        )

        self.block = nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.block(x)  # add skip connections
        return out

def create_spherical_grid(width, height, data_type=torch.float32):
    v_range = (
        torch.arange(0, height) # [0 - h]
        .view(1, height, 1) # [1, [0 - h], 1]
        .expand(1, height, width) # [1, [0 - h], W]
        .type(data_type)  # [1, H, W]
    )
    u_range = (
        torch.arange(0, width) # [0 - w]
        .view(1, 1, width) # [1, 1, [0 - w]]
        .expand(1, height, width) # [1, H, [0 - w]]
        .type(data_type)  # [1, H, W]
    )
    u_range *= (2 * numpy.pi / width) # [0, 2 * pi]
    v_range *= (numpy.pi / height) # [0, pi]
    return torch.stack((u_range, v_range), dim=1)  # [1, 2, H, W]

class AddCoords360(nn.Module):
    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoords360, self).__init__()
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        
        batch_size_tensor = input_tensor.shape[0]
        '''
        ret = create_spherical_grid(self.y_dim, self.x_dim).cuda()
        ret = ret.repeat(batch_size_tensor, 1, 1, 1)
        ret = torch.cat([input_tensor, ret], 1)
        '''
        

        xx_ones = torch.ones([1, self.y_dim], dtype=torch.float32, device=input_tensor.device)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.float32, device=input_tensor.device).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.float32, device=input_tensor.device)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.float32, device=input_tensor.device).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel.float() / (self.x_dim - 1)
        yy_channel = yy_channel.float() / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        
        return ret

class CoordConv360(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, in_channels, out_channels, kernel_size, *args, **kwargs):
        super(CoordConv360, self).__init__()
        self.addcoords = AddCoords360(x_dim=x_dim, y_dim=y_dim, with_r=with_r)                
        in_size = in_channels+2
        if with_r:
            in_size += 1            
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size, **kwargs)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret


def create_conv(in_size, out_size, conv_type, padding=1, stride=1, kernel_size=3, width=512):
    if conv_type == 'standard':
        return nn.Conv2d(in_channels=in_size, out_channels=out_size, \
            kernel_size=kernel_size, padding=padding, stride=stride)
    elif conv_type == 'coord':
        return CoordConv360(x_dim=width / 2.0, y_dim=width,\
            with_r=False, kernel_size=kernel_size, stride=stride,\
            in_channels=in_size, out_channels=out_size, padding=padding)    

def create_activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'elu':
        return nn.ELU(inplace=True)

class Identity(nn.Module):
    def forward(self, x):
        return x

def create_normalization(out_size, norm_type):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(out_size)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(out_size // 4, out_size)
    elif norm_type == 'none':
        return Identity()

def create_downscale(out_size, down_mode):
    if down_mode == 'pool':
        return torch.nn.modules.MaxPool2d(2)
    elif down_mode == 'downconv':
        return nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3,\
            stride=2, padding=1, bias=False)
    elif down_mode == 'gaussian':
        print("Not implemented")        

class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(inplanes, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(inplanes, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(256*5, outplanes, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(outplanes)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        return out