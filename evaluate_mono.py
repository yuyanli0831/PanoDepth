from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
import scipy.io
from metrics import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset_loader import OmniDepthDataset
import cv2
import spherical as S360
import supervision as L
from util import *
#import weight_init
from sync_batchnorm import convert_model
from CasStereoNet.models import psmnet_spherical
from CasStereoNet.models.loss import stereo_psmnet_loss
#from network_resnet import ResNet360
from network_extra_constraint import ResNet360
from network_rectnet import RectNet
import matplotlib.pyplot as plot
import scipy.io

parser = argparse.ArgumentParser(description='PanoDepth')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='psmnet',
                    help='select model')
parser.add_argument('--input_dir', default='/media/rtx2/DATA/Student_teacher_depth/stanford2d3d',
                    help='input data directory')
parser.add_argument('--trainfile', default='train_stanford2d3d.txt',
                    help='train file name')
parser.add_argument('--testfile', default='test_stanford2d3d.txt',
                    help='validation file name')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--start_decay', type=int, default=60,
                    help='number of epoch for lr to start decay')
parser.add_argument('--start_learn', type=int, default=100,
                    help='number of iterations for stereo network to start learn')
parser.add_argument('--batch', type=int, default=8,
                    help='number of batch to train')
parser.add_argument('--visualize_interval', type=int, default=20,
                    help='number of batch to train')
parser.add_argument('--baseline', type=float, default=0.24,
                    help='image pair baseline distance')
parser.add_argument('--interval', type=float, default=0.5,
                    help='second stage interval')
parser.add_argument('--nlabels', type=str, default="48,24", 
                    help='number of labels')
parser.add_argument('--checkpoint', default= None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='./checkpoints',
                    help='save checkpoint path')
parser.add_argument('--visualize_path', default='./visualize_extra_constraints',
                    help='save checkpoint path')                    
parser.add_argument('--tensorboard_path', default='./logs',
                    help='tensorboard path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--real', action='store_true', default=False,
                    help='adapt to real world images in both training and validation')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


#-----------------------------------------

# Random Seed -----------------------------
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#------------------------------------------
input_dir = args.input_dir # Dataset location
train_file_list = args.trainfile # File with list of training files
val_file_list = args.testfile # File with list of validation files

#------------------------------------
result_view_dir = args.visualize_path + '/evaluation'
if not os.path.exists(result_view_dir):
    os.makedirs(result_view_dir) 
   

val_dataset = OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=val_file_list)

val_dataloader = torch.utils.data.DataLoader(
	dataset=val_dataset,
	batch_size=1,
	shuffle=False,
	num_workers=8,
	drop_last=False)


#----------------------------------------------------------
#first network, coarse depth estimation
# option 1, resnet 360 
num_gpu = torch.cuda.device_count()
#first_network = ResNet360(wf=32, norm_type='batchnorm', activation='relu', aspp=False)

first_network = ResNet360()

#weight_init.initialize_weights(first_network, init="xavier", pred_bias=float(5.0))
#first_network = RectNet()
first_network = convert_model(first_network)
# option 2, spherical unet
#view_syn_network = SphericalUnet()
first_network = nn.DataParallel(first_network)
first_network.cuda()
#----------------------------------------------------------

state_dict = torch.load(result_view_dir.split('/')[1] + '/checkpoints/checkpoint_latest.tar')
first_network.load_state_dict(state_dict['state_dict'])


# Valid Function -----------------------
def val(rgb, depth, mask, batch_idx):
    
    mask = mask>0 
    with torch.no_grad():
        outputs, _ = first_network(rgb)
        #outputs = F.interpolate(outputs, size=[256, 512], mode='bilinear', align_corners=True)
        
    rgb = rgb[:,:3,:,:].detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()
    depth_prediction = outputs.detach().cpu().numpy()
    depth_prediction[depth_prediction>8] = 0
    rgb_img = rgb[0, :, :, :].transpose(1,2,0)
    depth_img = depth[0, 0, :, :]
    depth_pred_img = depth_prediction[0, 0, :, :]
    cv2.imwrite('{}/test_rgb_{}.png'.format(result_view_dir, batch_idx), rgb_img*255)
    plot.imsave('{}/test_depth_gt_{}.png'.format(result_view_dir, batch_idx), depth_img, cmap="jet")
    plot.imsave('{}/test_depth_pred_{}.png'.format(result_view_dir, batch_idx), depth_pred_img, cmap="jet")

    return outputs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {'val' : self.val,
            'sum' : self.sum,
            'count' : self.count,
            'avg' : self.avg}

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']

abs_rel_error_meter = AverageMeter()
sq_rel_error_meter = AverageMeter()
lin_rms_sq_error_meter = AverageMeter()
log_rms_sq_error_meter = AverageMeter()
d1_inlier_meter = AverageMeter()
d2_inlier_meter = AverageMeter()
d3_inlier_meter = AverageMeter()


def compute_eval_metrics(output, gt, depth_mask):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output
        gt_depth = gt

        N = depth_mask.sum()

        # Align the prediction scales via median
        median_scaling_factor = gt_depth[depth_mask>0].median() / depth_pred[depth_mask>0].median()
        depth_pred *= median_scaling_factor

        abs_rel = abs_rel_error(depth_pred, gt_depth, depth_mask)
        sq_rel = sq_rel_error(depth_pred, gt_depth, depth_mask)
        rms_sq_lin = lin_rms_sq_error(depth_pred, gt_depth, depth_mask)
        rms_sq_log = log_rms_sq_error(depth_pred, gt_depth, depth_mask)
        d1 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=1)
        d2 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=2)
        d3 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=3)
        
        abs_rel_error_meter.update(abs_rel, N)
        sq_rel_error_meter.update(sq_rel, N)
        lin_rms_sq_error_meter.update(rms_sq_lin, N)
        log_rms_sq_error_meter.update(rms_sq_log, N)
        d1_inlier_meter.update(d1, N)
        d2_inlier_meter.update(d2, N)
        d3_inlier_meter.update(d3, N)
 

# Main Function ---------------------------------------------------------------------------------------------
def main():
    first_network.eval()
    for batch_idx, (rgb, depth, depth_mask) in enumerate(tqdm(val_dataloader)):
        rgb, depth, depth_mask = rgb.cuda(), depth.cuda(), depth_mask.cuda()
        val_output = val(rgb, depth, depth_mask, batch_idx)
        
        compute_eval_metrics(val_output, depth, depth_mask)

        #------------
    print(
    '  Avg. Abs. Rel. Error: {:.4f}\n'
    '  Avg. Sq. Rel. Error: {:.4f}\n'
    '  Avg. Lin. RMS Error: {:.4f}\n'
    '  Avg. Log RMS Error: {:.4f}\n'
    '  Inlier D1: {:.4f}\n'
    '  Inlier D2: {:.4f}\n'
    '  Inlier D3: {:.4f}\n\n'.format(
    abs_rel_error_meter.avg,
    sq_rel_error_meter.avg,
    math.sqrt(lin_rms_sq_error_meter.avg),
    math.sqrt(log_rms_sq_error_meter.avg),
    d1_inlier_meter.avg,
    d2_inlier_meter.avg,
    d3_inlier_meter.avg))

if __name__ == '__main__':
    main()
        