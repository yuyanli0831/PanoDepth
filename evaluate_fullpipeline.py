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
from CasStereoNet.models import psmnet_spherical_up_down_inv_depth
from CasStereoNet.models.loss import stereo_psmnet_loss
#from network_resnet import ResNet360
from network_resnet_v2 import ResNet360
# network_rectnet import RectNet
import matplotlib.pyplot as plot
import scipy.io
import csv

parser = argparse.ArgumentParser(description='PanoDepth')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='psmnet',
                    help='select model')
parser.add_argument('--input_dir', default='/media/rtx2/DATA/Student_teacher_depth/stanford2d3d',
#parser.add_argument('--input_dir', default='/media/quadro/DATA2/omnidepth/data',
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
parser.add_argument('--visualize_path', default='./visualize_stanford2d3d_ours_48_24',
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
  
#-------------------------------------------------------------------
baseline = 0.24
direction = 'vertical'

batch_size = args.batch
maxdisp = args.maxdisp
nlabels = [int(nd) for nd in args.nlabels.split(",") if nd]
visualize_interval = args.visualize_interval
interval = args.interval

val_dataset = OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=val_file_list)

val_dataloader = torch.utils.data.DataLoader(
	dataset=val_dataset,
	batch_size=2,
	shuffle=False,
	num_workers=8,
	drop_last=True)


#----------------------------------------------------------
#first network, coarse depth estimation
# option 1, resnet 360 
num_gpu = torch.cuda.device_count()
first_network = ResNet360()

#first_network = ResNet360(conv_type='standard', norm_type='batchnorm', activation='relu', aspp=True)
#weight_init.initialize_weights(first_network, init="xavier", pred_bias=float(5.0))
#first_network = RectNet()
first_network = convert_model(first_network)
# option 2, spherical unet
#view_syn_network = SphericalUnet()
first_network = nn.DataParallel(first_network)
first_network.cuda()
#----------------------------------------------------------

stereo_network = psmnet_spherical_up_down_inv_depth.PSMNet(nlabels, 
        [1,0.5], True, 5, cr_base_chs=[32,32,16])
stereo_network = convert_model(stereo_network)
#-----------------------------------------------------------------------------
stereo_network = nn.DataParallel(stereo_network)
stereo_network.cuda()
#-------------------------------------------------

state_dict = torch.load(result_view_dir.split('/')[1] + '/checkpoints/checkpoint_latest.tar')
first_network.load_state_dict(state_dict['state_dict1'])
stereo_network.load_state_dict(state_dict['state_dict2'])


def val(target, render, depth, mask, batch_idx):
    stereo_network.eval() 

    with torch.no_grad():
        #outputs = stereo_network(render[0], target, baseline=baseline, direction=baseline_direction)
        outputs =stereo_network(render, target)
        output3 = outputs["stage2"]["pred"]
        sgrid = S360.grid.create_spherical_grid(512).cuda()
        #output3 = S360.derivatives.disparity_to_depth_vertical(sgrid, output3.unsqueeze(1), baseline).squeeze(1)

    gt = target[:,:3,:,:].detach().cpu().numpy()
    render_np = render.detach().cpu().numpy() 

    depth = depth.detach().cpu().numpy()
    sgrid = S360.grid.create_spherical_grid(512).cuda()
    depth_prediction = output3.detach().cpu().numpy()
    depth_prediction[depth_prediction>8] = 0
    if batch_idx % 10 == 0 and batch_idx > 0:
            gt_img = gt[0, :, :, :].transpose(1,2,0)
            depth_img = depth[0, :, :]
            #depth_down_img = cv2.normalize(depth_down_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            depth_pred_img = depth_prediction[0, :, :]
            #depth_pred_img = cv2.normalize(depth_pred_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite('{}/gt_{}.png'.format(result_view_dir, batch_idx), gt_img*255)
            #for num_render in range(len(render_np)):
            render_img = render_np[0, :, :, :].transpose(1,2,0)
            cv2.imwrite('{}/test_render_{}.png'.format(result_view_dir, batch_idx), render_img*255)
            plot.imsave('{}/test_depth_gt_{}.png'.format(result_view_dir, batch_idx), depth_img, cmap="jet")
            plot.imsave('{}/test_depth_pred_{}.png'.format(result_view_dir, batch_idx), depth_pred_img, cmap="jet")
    return output3

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

abs_rel_error_meter_coarse = AverageMeter()
sq_rel_error_meter_coarse = AverageMeter()
lin_rms_sq_error_meter_coarse = AverageMeter()
log_rms_sq_error_meter_coarse = AverageMeter()
d1_inlier_meter_coarse = AverageMeter()
d2_inlier_meter_coarse = AverageMeter()
d3_inlier_meter_coarse = AverageMeter()

def compute_eval_metrics2(output, gt, depth_mask):
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
 
def compute_eval_metrics1(output, gt, depth_mask):
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
        
        abs_rel_error_meter_coarse.update(abs_rel, N)
        sq_rel_error_meter_coarse.update(sq_rel, N)
        lin_rms_sq_error_meter_coarse.update(rms_sq_lin, N)
        log_rms_sq_error_meter_coarse.update(rms_sq_log, N)
        d1_inlier_meter_coarse.update(d1, N)
        d2_inlier_meter_coarse.update(d2, N)
        d3_inlier_meter_coarse.update(d3, N)


# Main Function ---------------------------------------------------------------------------------------------
def main():
    global_step = 0
    global_val = 0


    first_network.eval()
    for batch_idx, (rgb, depth, depth_mask) in tqdm(enumerate(val_dataloader)):
            sgrid = S360.grid.create_spherical_grid(512).cuda()
            uvgrid = S360.grid.create_image_grid(512, 256).cuda()
            rgb, depth, depth_mask = rgb.cuda(), depth.cuda(), (depth_mask>0).cuda()
            with torch.no_grad():
                coarse_depth_pred = first_network(rgb[:,:3,:,:])
            
            if direction == 'vertical':
                render = dibr_vertical(coarse_depth_pred.clamp(0.1, 8.0), rgb, uvgrid, sgrid, baseline=baseline)
            elif direction == 'horizontal':
                render = dibr_horizontal(coarse_depth_pred.clamp(0.1, 8.0), rgb, uvgrid, sgrid, baseline=baseline)
            else:
                raise NotImplementedError

            depth_np = coarse_depth_pred.detach().cpu().numpy()
            if batch_idx % 10 == 0 and batch_idx > 0:
                depth_coarse_img = depth_np[0, 0, :, :]
                depth_coarse_img[depth_coarse_img>8] = 0
                depth_coarse_img = (depth_coarse_img / 8 * 65535).astype(np.uint16)
                plot.imsave('{}/coarse_vis_{}.png'.format(
                   result_view_dir, batch_idx), depth_coarse_img, cmap='jet')

            val_output = val(rgb, render, depth.squeeze(1), depth_mask.squeeze(1), batch_idx)
            compute_eval_metrics1(coarse_depth_pred, depth, depth_mask)
            compute_eval_metrics2(val_output, depth.squeeze(1), depth_mask.squeeze(1))
	        #-------------------------------------------------------------
            # Loss ---------------------------------
            #total_val_loss += val_loss
            #---------------------------------------
            # Step ------
            global_val+=1
            #------------
        #writer.add_scalar('total validation loss',total_val_loss/(len(val_dataloader)),epoch) #tensorboardX for validation in epoch        
        #writer.add_scalar('total validation crop 26 depth rmse',total_val_crop_rmse/(len(val_dataloader)),epoch) #tensorboardX rmse for validation in epoch
    print(
        '  Avg. Abs. Rel. Error: coarse {:.4f}, final {:.4f}\n'
        '  Avg. Sq. Rel. Error: coarse {:.4f}, final {:.4f}\n'
        '  Avg. Lin. RMS Error: coarse {:.4f}, final {:.4f}\n'
        '  Avg. Log RMS Error: coarse {:.4f}, final {:.4f}\n'
        '  Inlier D1: coarse {:.4f}, final {:.4f}\n'
        '  Inlier D2: coarse {:.4f}, final {:.4f}\n'
        '  Inlier D3: coarse {:.4f}, final {:.4f}\n\n'.format(
    abs_rel_error_meter_coarse.avg,
    abs_rel_error_meter.avg,
    sq_rel_error_meter_coarse.avg,
    sq_rel_error_meter.avg,
    math.sqrt(lin_rms_sq_error_meter_coarse.avg),
    math.sqrt(lin_rms_sq_error_meter.avg),
    math.sqrt(log_rms_sq_error_meter_coarse.avg),
    math.sqrt(log_rms_sq_error_meter.avg),
    d1_inlier_meter_coarse.avg,
    d1_inlier_meter.avg,
    d2_inlier_meter_coarse.avg,
    d2_inlier_meter.avg,
    d3_inlier_meter_coarse.avg,
    d3_inlier_meter.avg))

    csv_filename = os.path.join(result_view_dir, 'final_result.csv')
    fields = ['Abs Rel', 'Sq Rel', 'Lin RMSE', 'log RMSE', 'D1', 'D2', 'D3']
    csvfile = open(csv_filename, 'w', newline='')
    with csvfile:
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields)
        row = ['{:.4f}'.format(abs_rel_error_meter_coarse.avg.item()), 
            '{:.4f}'.format(sq_rel_error_meter_coarse.avg.item()), 
            '{:.4f}'.format(torch.sqrt(lin_rms_sq_error_meter_coarse.avg).item()),
            '{:.4f}'.format(torch.sqrt(log_rms_sq_error_meter_coarse.avg).item()), 
            '{:.4f}'.format(d1_inlier_meter_coarse.avg.item()), 
            '{:.4f}'.format(d2_inlier_meter_coarse.avg.item()), 
            '{:.4f}'.format(d3_inlier_meter_coarse.avg.item())]

        csvwriter.writerow(row)
        row = ['{:.4f}'.format(abs_rel_error_meter.avg.item()), 
            '{:.4f}'.format(sq_rel_error_meter.avg.item()), 
            '{:.4f}'.format(torch.sqrt(lin_rms_sq_error_meter.avg).item()),
            '{:.4f}'.format(torch.sqrt(log_rms_sq_error_meter.avg).item()), 
            '{:.4f}'.format(d1_inlier_meter.avg.item()), 
            '{:.4f}'.format(d2_inlier_meter.avg.item()), 
            '{:.4f}'.format(d3_inlier_meter.avg.item())]

        csvwriter.writerow(row)


if __name__ == '__main__':
    main()
        