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
from network_syn_mapped import SphericalUnet
from network_resnet import ResNet360
from util import load_partial_model
import weight_init
from sync_batchnorm import convert_model
from CasStereoNet.models import psmnet_spherical_up_down_inv_depth_multi
from CasStereoNet.models.loss import stereo_psmnet_loss
import matplotlib.pyplot as plot

parser = argparse.ArgumentParser(description='360SD-Net')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='psmnet',
                    help='select model')
parser.add_argument('--input_dir', default='../360SD-net/data/Realistic',
                    help='input data directory')
parser.add_argument('--trainfile', default='train.txt',
                    help='train file name')
parser.add_argument('--testfile', default='test.txt',
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
parser.add_argument('--nlabels', type=str, default="48,24", 
                    help='number of labels')
parser.add_argument('--checkpoint', default= None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='./checkpoints',
                    help='save checkpoint path')
parser.add_argument('--visualize_path', default='./visualize_multi_stereo_3views_vertical',
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
batch_size = args.batch

#baseline to render images, len is the total number of rendered views
baseline = [0.24, -0.24]
baseline_direction = ['vertical', 'vertical']
maxdisp = args.maxdisp
nlabels = [int(nd) for nd in args.nlabels.split(",") if nd]
visualize_interval = args.visualize_interval
interval = args.interval
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
    stereo_network.load_state_dict(state_dict['state_dict'])
    start_epoch = state_dict['epoch']

print('## Batch size: {}'.format(batch_size))  
print('## Number of stereo matching model parameters: {}'.format(sum([p.data.nelement() for p in stereo_network.parameters()])))
#--------------------------------------------------

# Optimizer ----------
optimizer = optim.Adam(list(stereo_network.parameters()), 
        lr=0.001, betas=(0.9, 0.999))
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
def train(target, render, depth, mask, batch_idx):
    stereo_network.train()
    #mask = mask>0

    optimizer.zero_grad()
    # Loss -------------------------------------------- 

    outputs = stereo_network(render, target, baseline=baseline, direction=baseline_direction)

    dlossw = [0.5, 2.0]
    depth_loss = stereo_psmnet_loss(outputs, depth, mask, dlossw=[0.5, 2.0])
    #loss = disp_loss
    #--------------------------------------------------
    output3 = outputs["stage2"]["pred"]
    gt = target[:,:3,:,:].detach().cpu().numpy()
    render_np = [r.detach().cpu().numpy() for r in render]
    #render_np1 = render[0][:,:3,:,:].detach().cpu().numpy()
    #render_np2 = render[1][:,:3,:,:].detach().cpu().numpy()

    depth = depth.detach().cpu().numpy()
    sgrid = S360.grid.create_spherical_grid(512).cuda()
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

    return depth_loss

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
            #cv2.imwrite(result_view_dir + '/depth_gt_' + str(i) + '.png', depth_down_img)
            #cv2.imwrite(result_view_dir + '/depth_pred_' + str(i) + '.png', depth_pred_img)
            plot.imsave(result_view_dir + '/test_depth_pred_' + str(i) + '.png', depth_pred_img, cmap="viridis")
            plot.imsave(result_view_dir + '/test_depth_gt_' + str(i) + '.png', depth_down_img, cmap="viridis")

    return output3

# Adjust Learning Rate
def adjust_learning_rate(optimizer, epoch):
    
    lr = 0.001
    if epoch > args.start_decay:
        lr = 0.0005
     
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

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
    global_step = 0
    global_val = 0

    # Start Training ---------------------------------------------------------
    start_full_time = time.time()
    for epoch in tqdm(range(start_epoch+1, args.epochs+1), desc='Epoch'):
        print('---------------Train Epoch', epoch, '----------------')
        total_train_loss = 0
        stereo_matching_loss = 0
        adjust_learning_rate(optimizer, epoch)

 
        # Train --------------------------------------------------------------------------------------------------
        for batch_idx, (up, down, depth_up, depth_down, depth_upmask, depth_downmask) in enumerate(train_dataloader):
            #get ground truth up-down disparity
            sgrid = S360.grid.create_spherical_grid(512).cuda()
            uvgrid = S360.grid.create_image_grid(512, 256).cuda()
            depth = depth_down.cuda()
            depth_mask = depth_downmask.cuda()
            target = down.cuda()
            num_render_view = len(baseline)
            render = []
            for bl, direction in zip(baseline, baseline_direction):
                if direction == 'vertical':
                    render.append(dibr_vertical(depth, target, uvgrid, sgrid, baseline=bl))
                elif direction == 'horizontal':
                    render.append(dibr_horizontal(depth, target, uvgrid, sgrid, baseline=bl))
                else:
                    raise NotImplementedError
            #render = [dibr_vertical(depth, target, uvgrid, sgrid, baseline=bl) for bl in baseline]

            depth_loss = train(target, render, depth.squeeze(1), depth_mask, batch_idx)
            
            loss = depth_loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            
            global_step += 1
            if batch_idx % 20 == 0:
                print('[Epoch %d--Iter %d]total loss %.4f' % 
                (epoch, batch_idx, total_train_loss/(batch_idx+1)))
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
                'state_dict': stereo_network.state_dict(),
                'train_loss': total_train_loss/len(train_dataloader),
            }, savefilename)
        #-----------------------------------------------------------------------------

        # Valid ----------------------------------------------------------------------------------------------------
        total_val_loss = 0
        total_val_crop_rmse = 0
        print('-------------Validate Epoch', epoch, '-----------')
        for batch_idx, (up, down, depth_up, depth_down, depth_upmask, depth_downmask) in enumerate(val_dataloader):
            sgrid = S360.grid.create_spherical_grid(512).cuda()
            uvgrid = S360.grid.create_image_grid(512, 256).cuda()
            depth = depth_down.cuda()
            depth_mask = depth_downmask.cuda()
            target = down.cuda()
            #render1 = dibr_vertical(depth, target, uvgrid, sgrid, baseline=baseline[0])
            #render2 = dibr_vertical(depth, target, uvgrid, sgrid, baseline=baseline[1])
            #render = [render1, render2]
            num_render_view = len(baseline)
            render = []
            for bl, direction in zip(baseline, baseline_direction):
                if direction == 'vertical':
                    render.append(dibr_vertical(depth, target, uvgrid, sgrid, baseline=bl))
                elif direction == 'horizontal':
                    render.append(dibr_horizontal(depth, target, uvgrid, sgrid, baseline=bl))
                else:
                    raise NotImplementedError
            #render = [dibr_vertical(depth, target, uvgrid, sgrid, baseline=bl) for bl in baseline]
            val_output = val(target, render, depth.squeeze(1), depth_mask, batch_idx)

            compute_eval_metrics(val_output, depth.squeeze(1), depth_mask)
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
        '  Avg. Abs. Rel. Error: {:.4f}\n'
        '  Avg. Sq. Rel. Error: {:.4f}\n'
        '  Avg. Lin. RMS Error: {:.4f}\n'
        '  Avg. Log RMS Error: {:.4f}\n'
        '  Inlier D1: {:.4f}\n'
        '  Inlier D2: {:.4f}\n'
        '  Inlier D3: {:.4f}\n\n'.format(
        epoch, 
        abs_rel_error_meter.avg,
        sq_rel_error_meter.avg,
        math.sqrt(lin_rms_sq_error_meter.avg),
        math.sqrt(log_rms_sq_error_meter.avg),
        d1_inlier_meter.avg,
        d2_inlier_meter.avg,
        d3_inlier_meter.avg))
        writer.add_scalar('Avg Abs Rel', abs_rel_error_meter.avg, epoch+1)
        writer.add_scalar('Avg Sq Rel', sq_rel_error_meter.avg, epoch+1)
        writer.add_scalar('Avg. Lin. RMS', math.sqrt(lin_rms_sq_error_meter.avg), epoch+1)
        writer.add_scalar('Avg. Log. RMS', math.sqrt(log_rms_sq_error_meter.avg), epoch+1)
        writer.add_scalar('Inlier D1', d1_inlier_meter.avg, epoch+1)
        writer.add_scalar('Inlier D2', d2_inlier_meter.avg, epoch+1)
        writer.add_scalar('Inlier D3', d3_inlier_meter.avg, epoch+1)
    writer.close()
    # End Training
    print("Training Ended hahahaha!!!")
    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
#----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
        
