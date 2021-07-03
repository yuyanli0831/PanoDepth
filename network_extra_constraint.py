import numpy
import torch
import torch.nn as nn

import functools
import torch.nn.functional as F
import torchvision.models as models

# adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py 

def load_pretrain_resnet(name='resnet18', pretrained=False):
    if name == 'resnet34':
        return models.resnet34(pretrained=pretrained)
    if name == 'resnet18':
        return models.resnet18(pretrained=pretrained)
    if name == 'resnet50':
        return models.resnet50(pretrained=pretrained)

class UpProject(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpProject, self).__init__()


        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))
        #out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 0, 1)))#right interleaving padding
        out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github
        #out1_3 = self.conv1_3(nn.functional.pad(x, (0, 1, 1, 1)))#right interleaving padding
        out1_3 = self.conv1_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github
        #out1_4 = self.conv1_4(nn.functional.pad(x, (0, 1, 0, 1)))#right interleaving padding
        out1_4 = self.conv1_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))
        #out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 0, 1)))#right interleaving padding
        out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github
        #out2_3 = self.conv2_3(nn.functional.pad(x, (0, 1, 1, 1)))#right interleaving padding
        out2_3 = self.conv2_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github
        #out2_4 = self.conv2_4(nn.functional.pad(x, (0, 1, 0, 1)))#right interleaving padding
        out2_4 = self.conv2_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()     
        self.convA = nn.Sequential(   
                nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(output_features),
                nn.LeakyReLU(0.2)
        )
        self.convB = nn.Sequential(   
                nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(output_features),
                nn.LeakyReLU(0.2)
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        concat = torch.cat([up_x, concat_with], dim=1)
        convA = self.convA(concat)
        convB = self.convB(convA)
        return convB

class ResNet360(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=256,
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
        
        super(ResNet360, self).__init__()
        
        self.input0_0 = ConvBlock(in_channels, 16, (3,9), padding=(1,4))
        self.input0_1 = ConvBlock(in_channels, 16, (5,11), padding=(2,5))
        self.input0_2 = ConvBlock(in_channels, 16, (5,7), padding=(2,3))
        self.input0_3 = ConvBlock(in_channels, 16, 7, padding=3)
        
        self.encoder = load_pretrain_resnet('resnet34')
        self.conv1 = self.encoder.conv1
        self.bn1 = self.encoder.bn1
        self.relu = self.encoder.relu
        self.maxpool = self.encoder.maxpool
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        self.aspp = ASPP(512, features)
        #up sample
        #self.conv2 = nn.Conv2d(512, features, kernel_size=1, stride=1, padding=0)
        #self.bn = nn.BatchNorm2d(features)
        
        self.up1_1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2_1 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3_1 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4_1 = UpSample(skip_input=features//8 + 64,  output_features=features//16)
        self.conv_depth = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)
        
        self.up1_2 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2_2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3_2 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4_2 = UpSample(skip_input=features//8 + 64,  output_features=features//16)
        self.conv_normal = nn.Conv2d(features//16, 3, kernel_size=3, stride=1, padding=1)
        
        '''
        self.up1 = UpProject(256, 128)
        self.up2 = UpProject(128, 64)
        self.up3 = UpProject(64, 32)
        self.up4 = UpProject(32, 16)

        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        '''

    def forward(self, x):
        """Standard forward"""
    
        input0_0_out = self.input0_0(x)
        input0_1_out = self.input0_1(x)
        input0_2_out = self.input0_2(x)
        input0_3_out = self.input0_3(x)
        input0_out_cat = torch.cat(
            (input0_0_out, 
            input0_1_out, 
            input0_2_out, 
            input0_3_out), 1)
        
        
        encoder_features, decoder_features = [], []
        #x = self.relu(self.bn1(self.conv1(x)))
        x = torch.cat([input0_0_out, input0_1_out, input0_2_out, input0_3_out], 1)
        encoder_features.append(x)
        x = self.maxpool(x)       
        x = self.layer1(x)
        encoder_features.append(x)
        x = self.layer2(x)
        encoder_features.append(x)
        x = self.layer3(x)
        encoder_features.append(x)
        x = self.layer4(x)
        encoder_features.append(x)
              
       # x = F.relu(self.conv2(x))
        #x = self.bn2(x)
        encoder = self.aspp(x)
        x = self.up1_1(encoder, encoder_features[3])
        x = self.up2_1(x, encoder_features[2])
        x = self.up3_1(x, encoder_features[1])
        x = self.up4_1(x, encoder_features[0])

        depth = self.conv_depth(x)
        
        x = self.up1_2(encoder, encoder_features[3])
        x = self.up2_2(x, encoder_features[2])
        x = self.up3_2(x, encoder_features[1])
        x = self.up4_2(x, encoder_features[0])

        normal = self.conv_normal(x)
        
    
        return depth, normal


class ConvBlock(nn.Module):

    def __init__(self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding, 
                dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


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
