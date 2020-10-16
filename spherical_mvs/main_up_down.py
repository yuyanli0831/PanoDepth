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
from metrics import *
from torch.utils.tensorboard import SummaryWriter
from skimage.measure import compare_mse as mse
from tqdm import tqdm
from dataset_up_down import OmniDepthDataset
import cv2
import spherical as S360
import supervision as L
from util import load_partial_model
import weight_init
from sync_batchnorm import convert_model
from CasStereoNet.models import psmnet_spherical_up_down_inv_depth_multi
from CasStereoNet.models.loss import stereo_psmnet_loss
from network_resnet import ResNet360
import matplotlib.pyplot as plot

parser = argparse.ArgumentParser(description='360SD-Net')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='psmnet',
                    help='select model')
parser.add_argument('--input_dir', default='../360SD-net/data/Realistic',
                    help='input data directory')
parser.add_argument('--trainfile', default='train_full.txt',
                    help='train file name')
parser.add_argument('--testfile', default='test_full.txt',
                    help='validation file name')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--start_decay', type=int, default=30,
                    help='number of epoch for lr to start decay')
parser.add_argument('--start_learn', type=int, default=10,
                    help='number of epoch for LCV to start learn')
parser.add_argument('--batch', type=int, default=8,
                    help='number of batch to train')
parser.add_argument('--visualize_interval', type=int, default=60,
                    help='number of batch to train')
parser.add_argument('--baseline', type=float, default=0.24,
                    help='image pair baseline distance')
parser.add_argument('--interval', type=float, default=0.5,
                    help='second stage interval')
parser.add_argument('--nlabels', type=str, default="64,32", 
                    help='number of labels')
parser.add_argument('--checkpoint', default= None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='./checkpoints',
                    help='save checkpoint path')
parser.add_argument('--visualize_path', default='./visualize_mvs_larger_baseline',
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

# tensorboard Path -----------------------
writer_path = args.tensorboard_path
writer = SummaryWriter(writer_path)

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
result_view_dir = args.visualize_path
if not os.path.exists(result_view_dir):
    os.makedirs(result_view_dir)  
#-------------------------------------------------------------------
baseline = [0.4, -0.4]
baseline_direction = ['vertical', 'vertical']
batch_size = args.batch
maxdisp = args.maxdisp
nlabels = [int(nd) for nd in args.nlabels.split(",") if nd]
visualize_interval = args.visualize_interval
interval = args.interval
lr1 = 0.0002
lr2 = 0.0005
#-------------------------------------------------------------------
#data loaders
train_dataloader = torch.utils.data.DataLoader(
	dataset=OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=train_file_list),
	batch_size=batch_size,
	shuffle=True,
	num_workers=8,
	drop_last=True)

val_dataloader = torch.utils.data.DataLoader(
	dataset=OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=val_file_list),
	batch_size=batch_size,
	shuffle=False,
	num_workers=4,
	drop_last=True)

#----------------------------------------------------------
#first network, coarse depth estimation
# option 1, resnet 360 
first_network = ResNet360(norm_type='batchnorm', activation='relu')
weight_init.initialize_weights(first_network, init="xavier", pred_bias=float(5.0))
first_network = convert_model(first_network)
# option 2, spherical unet
#view_syn_network = SphericalUnet()

first_network = nn.DataParallel(first_network)
first_network.cuda()
#----------------------------------------------------------

# stereo matching network ----------------------------------------------
if args.model == 'psmnet':
    stereo_network = psmnet_spherical_up_down_inv_depth_multi.PSMNet(nlabels, 
        [1, interval], True, 5, cr_base_chs=[32,32,16])
    #convert bn to synchronized bn
    stereo_network = convert_model(stereo_network)

else:
    print('Model Not Implemented!!!')
#----------------------------------------------------------

#-----------------------------------------------------------------------------

# Multi_GPU for model ----------------------------
stereo_network = nn.DataParallel(stereo_network)
stereo_network.cuda()
#-------------------------------------------------

# Load Checkpoint -------------------------------
start_epoch = 0
if args.checkpoint is not None:
    state_dict = torch.load(args.checkpoint)
    first_network.load_state_dict(state_dict['state_dict1'])
    stereo_network.load_state_dict(state_dict['state_dict2'])
    start_epoch = state_dict['epoch']

print('## Batch size: {}'.format(batch_size))  
print('## learning rate 1: {}, learning rate 2: {}'.format(lr1, lr2))  
print('## Number of first model parameters: {}'.format(sum([p.data.nelement() for p in first_network.parameters()])))
print('## Number of stereo matching model parameters: {}'.format(sum([p.data.nelement() for p in stereo_network.parameters()])))
#--------------------------------------------------

# Optimizer ----------
#optimizer = optim.Adam(list(model.parameters())+list(view_syn_network.parameters()), 
#        lr=0.001, betas=(0.9, 0.999))
optimizer1 = optim.Adam(list(first_network.parameters()), 
        lr=lr1, betas=(0.9, 0.999))
optimizer2 = optim.Adam(list(stereo_network.parameters()), 
        lr=lr2, betas=(0.9, 0.999))
#---------------------

# Freeze Unfreeze Function 
# freeze_layer ----------------------
def freeze_layer(layer):
	for param in layer.parameters():
		param.requires_grad = False
# Unfreeze_layer --------------------
def unfreeze_layer(layer):
	for param in layer.parameters():
		param.requires_grad = True
 
def dibr_vertical(depth, image, uvgrid, sgrid, baseline):
    disp = torch.cat(
                (
                    torch.zeros_like(depth),
                    S360.derivatives.dtheta_vertical(sgrid, depth, baseline)
                ),
                dim=1
            )
    render_coords = uvgrid + disp
    render_coords[torch.isnan(render_coords)] = 0
    render_coords[torch.isinf(render_coords)] = 0
    rendered,_ = L.splatting.render(image, depth, render_coords, max_depth=8)
    return rendered

def dibr_horizontal(depth, image, uvgrid, sgrid, baseline):
    disp = torch.cat(
                (
                    S360.derivatives.dphi_horizontal_clip(sgrid, depth, baseline),
                    S360.derivatives.dtheta_horizontal_clip(sgrid, depth, baseline)
                ),
                dim=1
            )
    render_coords = uvgrid + disp
    render_coords[:, 0, :, :] = torch.fmod(render_coords[:, 0, :, :] + 512, 512)
    render_coords[torch.isnan(render_coords)] = 0
    render_coords[torch.isinf(render_coords)] = 0
    rendered, _ = L.splatting.render(image, depth, render_coords, max_depth=8)
    return rendered
    
# Train Function -------------------
def train(target, render, depth_down, mask, batch_idx):
    stereo_network.train()
    #mask = mask>0

    # mask value
    #mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask = mask>0
    mask.detach_()

    optimizer2.zero_grad()
    # Loss -------------------------------------------- 
    outputs = stereo_network(render, target, baseline=baseline, direction=baseline_direction)
    dlossw = [0.5, 2.0]
    
    stereo_loss = stereo_psmnet_loss(outputs, depth_down, mask, dlossw=[0.5, 2.0])
    #loss = disp_loss
    #--------------------------------------------------
    output3 = outputs["stage2"]["pred"]
    gt = target[:,:3,:,:].detach().cpu().numpy()
    render_np = [r.detach().cpu().numpy() for r in render]
    depth = depth_down.detach().cpu().numpy()
    depth_prediction = output3.detach().cpu().numpy()
    depth_prediction[depth_prediction>8] = 0

    if batch_idx % visualize_interval == 0:
        for i in range(2):
            gt_img = gt[i, :, :, :].transpose(1,2,0)
            depth_down_img = depth[i, :, :]
            #depth_down_img = cv2.normalize(depth_down_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            depth_pred_img = depth_prediction[i, :, :]
            #depth_pred_img = cv2.normalize(depth_pred_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite(result_view_dir + '/gt_' + str(i) + '.png', gt_img*255)
            #cv2.imwrite(result_view_dir + '/render1_' + str(i) + '.png', render_img1*255)
            #cv2.imwrite(result_view_dir + '/render2_' + str(i) + '.png', render_img2*255)
            for num_render in range(len(render_np)):
                render_img = render_np[num_render][i, :, :, :].transpose(1,2,0)
                cv2.imwrite(result_view_dir + '/render' + str(num_render) + '_'+ str(i) + '.png', render_img*255)
            plot.imsave(result_view_dir + '/depth_pred_' + str(i) + '.png', depth_pred_img, cmap="viridis")
            plot.imsave(result_view_dir + '/depth_gt_' + str(i) + '.png', depth_down_img, cmap="viridis")
            plot.imsave(result_view_dir + '/error_final_' + str(i) + '.png', abs(depth_down_img-depth_pred_img), cmap="Greys")


    return stereo_loss

# Valid Function -----------------------
def val(target, render, depth, mask, batch_idx):
    stereo_network.eval() 

    with torch.no_grad():
        outputs = stereo_network(render, target, baseline=baseline, direction=baseline_direction)
        output3 = outputs["stage2"]["pred"]

    gt = target[:,:3,:,:].detach().cpu().numpy()
    render_np = [r.detach().cpu().numpy() for r in render]

    depth = depth.detach().cpu().numpy()
    sgrid = S360.grid.create_spherical_grid(512).cuda()
    depth_prediction = output3.detach().cpu().numpy()
    depth_prediction[depth_prediction>8] = 0
    if batch_idx % visualize_interval == 0:
        for i in range(2):
            gt_img = gt[i, :, :, :].transpose(1,2,0)
            #render_img1 = render_np1[i, :, :, :].transpose(1,2,0)
            #render_img2 = render_np2[i, :, :, :].transpose(1,2,0)
            depth_down_img = depth[i, :, :]
            #depth_down_img = cv2.normalize(depth_down_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            depth_pred_img = depth_prediction[i, :, :]
            #depth_pred_img = cv2.normalize(depth_pred_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite(result_view_dir + '/test_gt_' + str(i) + '.png', gt_img*255)
            for num_render in range(len(render_np)):
                render_img = render_np[num_render][i, :, :, :].transpose(1,2,0)
                cv2.imwrite(result_view_dir + '/test_render' + str(num_render) + '_'+ str(i) + '.png', render_img*255)
            
            plot.imsave(result_view_dir + '/test_depth_pred_' + str(i) + '.png', depth_pred_img, cmap="viridis")
            plot.imsave(result_view_dir + '/test_depth_gt_' + str(i) + '.png', depth_down_img, cmap="viridis")
            plot.imsave(result_view_dir + '/test_error_final_' + str(i) + '.png', abs(depth_pred_img-depth_down_img), cmap="Greys")

    return output3

# Adjust Learning Rate
def adjust_learning_rate(optimizer1, optimizer2, epoch):
    
    lr1 = 0.0002
    lr2 = 0.0005
    if epoch > args.start_decay:
        lr1 = 0.0001
        lr2 = 0.0002
     
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr1
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr2

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

def compute_eval_metrics2(output1, gt, depth_mask):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output1
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

def compute_eval_metrics1(output2, gt, depth_mask):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output2
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

    # Start Training ---------------------------------------------------------
    start_full_time = time.time()
    for epoch in tqdm(range(start_epoch+1, args.epochs+1), desc='Epoch'):
        print('---------------Train Epoch', epoch, '----------------')
        total_train_loss = 0
        first_depth_estimation_loss = 0
        stereo_matching_loss = 0
        adjust_learning_rate(optimizer1, optimizer2, epoch)

        #-------------------------------
        first_network.train()
        # Train --------------------------------------------------------------------------------------------------
        for batch_idx, (up, down, depth_up, depth_down, depth_upmask, depth_downmask) in enumerate(train_dataloader):
            #get ground truth up-down disparity
            sgrid = S360.grid.create_spherical_grid(512).cuda()
            uvgrid = S360.grid.create_image_grid(512, 256).cuda()
            depth_down = depth_down.cuda()
            
            up, down = up.cuda(), down.cuda()
            depth_downmask = depth_downmask.cuda()
            #first depth estimation
            optimizer1.zero_grad()
            down_depth_pred = torch.abs(first_network(down[:,:3,:,:]))     
            attention_weights = S360.weights.theta_confidence(sgrid)
            # attention_weights = torch.ones_like(left_depth)
            downmask = depth_downmask > 0
            downmask = downmask.unsqueeze(1)
            # berhu depth loss with coordinates attention weight 
            depth_loss = L.direct.calculate_berhu_loss(down_depth_pred, depth_down,                
               mask=downmask, weights=attention_weights)
            left_xyz = S360.cartesian.coords_3d(sgrid, down_depth_pred)
            dI_dxyz = S360.derivatives.dV_dxyz(left_xyz)               
            guidance_duv = S360.derivatives.dI_duv(down[:,:3,:,:])
            depth_smoothness_loss = L.smoothness.guided_smoothness_loss(
                dI_dxyz, guidance_duv, downmask, (1.0 - attention_weights)
                * downmask.type(attention_weights.dtype)
            )
            depth_estimation_loss = depth_loss + depth_smoothness_loss * 0.1
            
            depth_gt_np = depth_down.detach().cpu().numpy()
            down_depth_np = down_depth_pred.detach().cpu().numpy()
            if batch_idx % visualize_interval == 0:
                for i in range(2):
                    depth_gt_img = depth_gt_np[i, 0, :, :]
                    down_depth_img = down_depth_np[i, 0, :, :]
                    down_depth_img[down_depth_img>8] = 0
                    plot.imsave(result_view_dir + '/coarse_depth_' + str(i) + '.png', down_depth_img, cmap="viridis")
                    plot.imsave(result_view_dir + '/error_coarse_' + str(i) + '.png', abs(down_depth_img-depth_gt_img), cmap="Greys")

            num_render_view = len(baseline)
            render = []
            for bl, direction in zip(baseline, baseline_direction):
                if direction == 'vertical':
                    render.append(dibr_vertical(down_depth_pred, down, uvgrid, sgrid, baseline=bl))
                elif direction == 'horizontal':
                    render.append(dibr_horizontal(down_depth_pred, down, uvgrid, sgrid, baseline=bl))
                else:
                    raise NotImplementedError

            stereo_loss = train(down, render, depth_down.squeeze(1), depth_downmask, batch_idx)
            
            #need to balance the first network loss and second stereo matching network loss
            alpha, beta = 0.01, 1
            if epoch == 1 and batch_idx<200:
                for param in stereo_network.parameters():
                    param.requires_grad = False
                loss = depth_estimation_loss
                stereo_loss = torch.tensor(0, dtype=torch.float32)
            else:
                for param in stereo_network.parameters():
                    param.requires_grad = True
                    
                loss = stereo_loss*alpha + depth_estimation_loss*beta
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            total_train_loss += loss.item()
            first_depth_estimation_loss += depth_estimation_loss.item() * beta
            stereo_matching_loss += stereo_loss.item() * alpha
            global_step += 1
            if batch_idx % 20 == 0:
                print('[Epoch %d--Iter %d]total loss %.4f, first depth loss %.4f, stereo loss %.4f' % 
                (epoch, batch_idx, total_train_loss/(batch_idx+1), first_depth_estimation_loss/(batch_idx+1), stereo_matching_loss/(batch_idx+1)))
            writer.add_scalar('first network depth loss', depth_estimation_loss, global_step)
            writer.add_scalar('stereo matching loss', stereo_loss, global_step)
            writer.add_scalar('total loss', loss, global_step) # tensorboardX for iter
        writer.add_scalar('total train loss',total_train_loss/len(train_dataloader),epoch) # tensorboardX for epoch
        #---------------------------------------------------------------------------------------------------------

        # Save Checkpoint -------------------------------------------------------------
        if not os.path.isdir(args.save_checkpoint):
            os.makedirs(args.save_checkpoint)
        if args.save_checkpoint[-1] == '/':
            args.save_checkpoint = args.save_checkpoint[:-1]
        savefilename = args.save_checkpoint+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
                'epoch': epoch,
                'state_dict1': first_network.state_dict(),
                'state_dict2': stereo_network.state_dict(),
                'train_loss': total_train_loss/len(train_dataloader),
            }, savefilename)
        #-----------------------------------------------------------------------------

        # Valid ----------------------------------------------------------------------------------------------------
        total_val_loss = 0
        total_val_crop_rmse = 0
        print('-------------Validate Epoch', epoch, '-----------')
        first_network.eval()
        for batch_idx, (up, down, depth_up, depth_down, depth_upmask, depth_downmask) in enumerate(val_dataloader):
            sgrid = S360.grid.create_spherical_grid(512).cuda()
            uvgrid = S360.grid.create_image_grid(512, 256).cuda()
            depth_down = depth_down.cuda()
            depth_downmask = depth_downmask.cuda()
            up, down = up.cuda(), down.cuda()
            with torch.no_grad():
                down_depth_pred = torch.abs(first_network(down[:,:3,:,:]))
            
            num_render_view = len(baseline)
            render = []
            for bl, direction in zip(baseline, baseline_direction):
                if direction == 'vertical':
                    render.append(dibr_vertical(down_depth_pred, down, uvgrid, sgrid, baseline=bl))
                elif direction == 'horizontal':
                    render.append(dibr_horizontal(down_depth_pred, down, uvgrid, sgrid, baseline=bl))
                else:
                    raise NotImplementedError
            
            depth_gt_np = depth_down.detach().cpu().numpy()
            down_depth_np = down_depth_pred.detach().cpu().numpy()
            if batch_idx % visualize_interval == 0:
                for i in range(2):
                    depth_gt_img = depth_gt_np[i, 0, :, :]
                    down_depth_img = down_depth_np[i, 0, :, :]
                    down_depth_img[down_depth_img>8] = 0
                    plot.imsave(result_view_dir + '/test_coarse_depth_' + str(i) + '.png', down_depth_img, cmap="viridis")
                    plot.imsave(result_view_dir + '/test_error_coarse_' + str(i) + '.png', abs(down_depth_img-depth_gt_img), cmap="Greys")
            
            val_output = val(down, render, depth_down.squeeze(1), depth_downmask, batch_idx)
            compute_eval_metrics1(down_depth_pred.squeeze(1), depth_down.squeeze(1), depth_downmask)
            compute_eval_metrics2(val_output, depth_down.squeeze(1), depth_downmask)
	        #-------------------------------------------------------------
            # Loss ---------------------------------
            #total_val_loss += val_loss
            #---------------------------------------
            # Step ------
            global_val+=1
            #------------
        #writer.add_scalar('total validation loss',total_val_loss/(len(val_dataloader)),epoch) #tensorboardX for validation in epoch        
        #writer.add_scalar('total validation crop 26 depth rmse',total_val_crop_rmse/(len(val_dataloader)),epoch) #tensorboardX rmse for validation in epoch
        print('Epoch: {}\n'
        '  Avg. Abs. Rel. Error: coarse {:.4f}, final {:.4f}\n'
        '  Avg. Sq. Rel. Error: coarse {:.4f}, final {:.4f}\n'
        '  Avg. Lin. RMS Error: coarse {:.4f}, final {:.4f}\n'
        '  Avg. Log RMS Error: coarse {:.4f}, final {:.4f}\n'
        '  Inlier D1: coarse {:.4f}, final {:.4f}\n'
        '  Inlier D2: coarse {:.4f}, final {:.4f}\n'
        '  Inlier D3: coarse {:.4f}, final {:.4f}\n\n'.format(
        epoch, 
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
        writer.add_scalars('Avg Abs Rel', {
             'final': abs_rel_error_meter.avg,
             'coarse': abs_rel_error_meter_coarse.avg,
             }, epoch+1)
        writer.add_scalars('Avg Sq Rel', {
             'final': sq_rel_error_meter.avg,
             'coarse': sq_rel_error_meter_coarse.avg,
             }, epoch+1)
        writer.add_scalars('Avg. Lin. RMS', {
             'final': math.sqrt(lin_rms_sq_error_meter.avg),
             'coarse': math.sqrt(lin_rms_sq_error_meter_coarse.avg),
             }, epoch+1)
        writer.add_scalars('Avg. Log. RMS', {
             'final': math.sqrt(log_rms_sq_error_meter.avg),
             'coarse': math.sqrt(log_rms_sq_error_meter_coarse.avg),
             }, epoch+1)
        writer.add_scalars('Inlier D1', {
             'final': d1_inlier_meter.avg,
             'coarse': d1_inlier_meter_coarse.avg,
             }, epoch+1)
        writer.add_scalars('Inlier D2', {
             'final': d2_inlier_meter.avg,
             'coarse': d2_inlier_meter_coarse.avg,
             }, epoch+1)
        writer.add_scalars('Inlier D3', {
             'final': d3_inlier_meter.avg,
             'coarse': d3_inlier_meter_coarse.avg,
             }, epoch+1)
        '''     
        writer.add_scalar('Avg Abs Rel', abs_rel_error_meter.avg, epoch+1)
        writer.add_scalar('Avg Sq Rel', sq_rel_error_meter.avg, epoch+1)
        writer.add_scalar('Avg. Lin. RMS', math.sqrt(lin_rms_sq_error_meter.avg), epoch+1)
        writer.add_scalar('Avg. Log. RMS', math.sqrt(log_rms_sq_error_meter.avg), epoch+1)
        writer.add_scalar('Inlier D1', d1_inlier_meter.avg, epoch+1)
        writer.add_scalar('Inlier D2', d2_inlier_meter.avg, epoch+1)
        writer.add_scalar('Inlier D3', d3_inlier_meter.avg, epoch+1)
        '''
    writer.close()
    # End Training
    print("Training Ended hahahaha!!!")
    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
#----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
        