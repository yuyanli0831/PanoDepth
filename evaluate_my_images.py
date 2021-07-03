import os
import sys
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import glob
from network_resnet_v2 import ResNet360
from CasStereoNet.models import psmnet_spherical
#from network_rectnet import RectNet
import cv2
import numpy as np
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
import spherical as S360
from util import *

num_gpu = torch.cuda.device_count()
#first_network = ResNet360(wf=32, norm_type='batchnorm', activation='relu', aspp=False)
first_network = ResNet360(2//num_gpu, output_size=(256, 512), aspp=True)
first_network = convert_model(first_network)
first_network = nn.DataParallel(first_network)
first_network.cuda()

stereo_network = psmnet_spherical.PSMNet(64, [48,24], 
        [2, 1], True, 5, cr_base_chs=[32,32,16])
stereo_network = convert_model(stereo_network)
stereo_network = nn.DataParallel(stereo_network)
stereo_network.cuda()

state_dict = torch.load('visualize_omnidepth_pretrain/checkpoint_latest.tar')
first_network.load_state_dict(state_dict['state_dict1'])
stereo_network.load_state_dict(state_dict['state_dict2'])
baseline = 0.24
direction = 'vertical'
'''
from bifuse_models.FCRN import MyModel as ResNet
model = ResNet(
    		layers=50,
    		decoder="upproj",
    		output_size=None,
    		in_channels=3,
    		pretrained=True
    		).cuda()
params = torch.load('BiFuse_Pretrained.pkl')
model.load_state_dict(params, strict=False)
model = model.eval()
'''
first_network = first_network.eval()
stereo_network = stereo_network.eval() 
myfiles = glob.glob('/home/quadro/Desktop/360indoor/*.jpg')
myfiles = np.sort(myfiles)
for filename in myfiles:
    img = cv2.imread(filename).astype(np.float32) / 255
    img = cv2.resize(img, (512, 256), interpolation = cv2.INTER_AREA)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = np.repeat(img, 2, axis=0)
    img = torch.from_numpy(img).float().cuda()
    render = []
    sgrid = S360.grid.create_spherical_grid(512).cuda()
    uvgrid = S360.grid.create_image_grid(512, 256).cuda()
    with torch.no_grad():
        coarse_depth_pred = torch.abs(first_network(img))
            
        if direction == 'vertical':
            render = dibr_vertical(coarse_depth_pred.clamp(0.1, 8.0), img, uvgrid, sgrid, baseline=baseline)
        elif direction == 'horizontal':
            render = dibr_horizontal(coarse_depth_pred.clamp(0.1, 8.0), img, uvgrid, sgrid, baseline=baseline)
        else:
            raise NotImplementedError

        depth_np = coarse_depth_pred.detach().cpu().numpy()

        depth_coarse_img = depth_np[0, 0, :, :]
        depth_coarse_img[depth_coarse_img>8] = 0
        #depth_coarse_img = (depth_coarse_img / 8 * 65535).astype(np.uint16)
        plot.imsave('/home/quadro/Desktop/360indoor/results_final/coarse_depth_vis_' + filename.split('/')[-1], 
           depth_coarse_img, cmap="jet")
    
    with torch.no_grad():
        outputs = stereo_network(render, img)
        output3 = outputs["stage2"]["pred"]
        output3 = S360.derivatives.disparity_to_depth_vertical(sgrid, output3.unsqueeze(1), baseline).squeeze(1)
    
        #raw_pred_var, pred_cube_var, refine = model(img)
        ### Convert to Numpy and Normalize to 0~1 ###
        #depth = torch.clamp(refine, 0, 8)
        #depth = F.interpolate(depth, scale_factor=0.5)
        depth = output3[0, :, :]
        depth_np = depth.detach().cpu().numpy()
        depth_np[depth_np>8] = 0

    plot.imsave('/home/quadro/Desktop/360indoor/results_final/depth_vis_' + filename.split('/')[-1], depth_np, cmap="jet")
    depth_np = (depth_np / 8 * 65535).astype(np.uint16)
    cv2.imwrite('/home/quadro/Desktop/360indoor/results_final/depth_' + filename.split('/')[-1], depth_np)
