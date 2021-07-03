from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from metrics import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset_loader import OmniDepthDataset
import cv2
from util import *
import spherical as S360
import supervision as L
#from network_syn_mapped import SphericalUnet
#from network_resnet import ResNet360
#from network_rectnet import RectNet
from network_resnet_v2 import ResNet360
from util import load_partial_model
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
import csv

parser = argparse.ArgumentParser(description='PanoDepth')
parser.add_argument('--model', default='psmnet',
                    help='select model')
parser.add_argument('--input_dir', default='/media/rtx2/DATA/Student_teacher_depth/stanford2d3d',
                    help='input data directory')
parser.add_argument('--trainfile', default='train_stanford2d3d.txt',
                    help='train file name')
parser.add_argument('--testfile', default='test_stanford2d3d.txt',
                    help='validation file name')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--start_decay', type=int, default=30,
                    help='number of epoch for lr to start decay')
parser.add_argument('--batch', type=int, default=16,
                    help='number of batch to train')
parser.add_argument('--visualize_interval', type=int, default=40,
                    help='number of batch to train')
parser.add_argument('--checkpoint', default= None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='./checkpoints',
                    help='save checkpoint path')
parser.add_argument('--visualize_path', default='./visualize_coarse',
                    help='save checkpoint path')                    
parser.add_argument('--tensorboard_path', default='logs',
                    help='tensorboard path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Save Checkpoint -------------------------------------------------------------
if not os.path.isdir(os.path.join(args.visualize_path, args.save_checkpoint)):
    os.makedirs(os.path.join(args.visualize_path, args.save_checkpoint))
# tensorboard Path -----------------------
writer_path = os.path.join(args.visualize_path,args.tensorboard_path)
if not os.path.isdir(writer_path):
    os.makedirs(writer_path)
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
visualize_interval = args.visualize_interval
lr = 2e-4

#-------------------------------------------------------------------
#data loaders
train_dataset = OmniDepthDataset(
		root_path=input_dir, 
        rotate=True,
        flip=True,
        gamma=False,
		path_to_img_list=train_file_list)

train_dataloader = torch.utils.data.DataLoader(
	dataset=train_dataset,
	batch_size=batch_size,
	shuffle=True,
	num_workers=8,
    pin_memory=True,
	drop_last=False)

val_dataset = OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=val_file_list)

val_dataloader = torch.utils.data.DataLoader(
	dataset=val_dataset,
	batch_size=4,
	shuffle=False,
	num_workers=8,
	drop_last=True)

#----------------------------------------------------------
#first network, coarse depth estimation
# option 1, resnet 360 
#first_network = ResNet360(conv_type='coord',norm_type='batchnorm', activation='relu', aspp=True)
num_gpu = torch.cuda.device_count()

#first_network = ResNet360(batch_size//num_gpu, output_size=(256, 512), aspp=True)
#first_network = RectNet()
first_network = ResNet360()
#weight_init.initialize_weights(first_network, init="xavier", pred_bias=float(5.0))

first_network = convert_model(first_network)

# parallel on multi gpu
first_network = nn.DataParallel(first_network)
first_network = first_network.cuda()

    

print('## Batch size: {}'.format(batch_size))  
print('## learning rate: {}'.format(lr))  
print('## Number of first model parameters: {}'.format(sum([p.data.nelement() for p in first_network.parameters()])))
#--------------------------------------------------

# Optimizer, and learning rate scheduler ----------
optimizer = optim.Adam(list(first_network.parameters()), 
        lr=lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.5)
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
 

# Train Function -------------------
def train(rgb, depth, mask, batch_idx):
    #mask = mask>0

    # mask value
    #mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask = mask>0
    mask.detach_()

    optimizer.zero_grad()
    # Loss -------------------------------------------- 
    outputs = first_network(rgb)
    #outputs = F.interpolate(outputs, size=[256, 512], mode='bilinear', align_corners=True)
    sgrid = S360.grid.create_spherical_grid(512).cuda()
    attention_weights = S360.weights.theta_confidence(sgrid)
    # attention_weights = torch.ones_like(left_depth)
    # berhu depth loss with coordinates attention weight 
    depth_loss = L.direct.calculate_berhu_loss(outputs, depth,                
               mask=mask, weights=attention_weights)
    left_xyz = S360.cartesian.coords_3d(sgrid, outputs)
    dI_dxyz = S360.derivatives.dV_dxyz(left_xyz)               
    guidance_duv = S360.derivatives.dI_duv(rgb)
    depth_smoothness_loss = L.smoothness.guided_smoothness_loss(
                dI_dxyz, guidance_duv, mask, (1.0 - attention_weights)
                *mask.type(attention_weights.dtype)
            )

    loss = depth_loss #+ 0.1*depth_smoothness_loss        
    #--------------------------------------------------
    rgb = rgb[:,:3,:,:].detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()
    depth_prediction = outputs.detach().cpu().numpy()
    depth_prediction[depth_prediction>8] = 0
    if batch_idx % visualize_interval == 0 and batch_idx > 0:
            rgb_img = rgb[0, :, :, :].transpose(1,2,0)
            depth_img = depth[0, 0, :, :]
            depth_pred_img = depth_prediction[0, 0, :, :]
            cv2.imwrite('{}/rgb_{}.png'.format(result_view_dir, batch_idx), rgb_img*255)
            plot.imsave('{}/depth_gt_{}.png'.format(result_view_dir, batch_idx), depth_img, cmap="jet")
            plot.imsave('{}/depth_pred_{}.png'.format(result_view_dir, batch_idx), depth_pred_img, cmap="jet")

    return loss

# Valid Function -----------------------
def val(rgb, depth, mask, batch_idx):
    
    mask = mask>0 
    with torch.no_grad():
        outputs = first_network(rgb)
        #outputs = F.interpolate(outputs, size=[256, 512], mode='bilinear', align_corners=True)
        
    rgb = rgb[:,:3,:,:].detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()
    depth_prediction = outputs.detach().cpu().numpy()
    depth_prediction[depth_prediction>8] = 0
    if batch_idx % 40 == 0:
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
    global_step = 0
    global_val = 0
    csv_filename = os.path.join(result_view_dir, 'logs/result_log.csv')
    fields = ['epoch', 'Abs Rel', 'Sq Rel', 'Lin RMSE', 'log RMSE', 'D1', 'D2', 'D3']
    csvfile = open(csv_filename, 'w', newline='')
    with csvfile:
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields) 
    # Start Training ---------------------------------------------------------
    start_full_time = time.time()
    for epoch in range(1, args.epochs+1):
        print('---------------Train Epoch', epoch, '----------------')
        total_train_loss = 0
        
        #adjust_learning_rate(optimizer, epoch)
        #-------------------------------
        first_network.train()
        # Train --------------------------------------------------------------------------------------------------
        for batch_idx, (rgb, depth, depth_mask) in tqdm(enumerate(train_dataloader),desc='it'):
 
            rgb, depth, depth_mask = rgb.cuda(), depth.cuda(), depth_mask.cuda()

            loss = train(rgb, depth, depth_mask, batch_idx)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            global_step += 1
            if batch_idx % 100 == 0 and batch_idx > 0:
                print('[Epoch %d--Iter %d]total loss %.4f' % 
                (epoch, batch_idx, total_train_loss/(batch_idx+1)))

            writer.add_scalar('total loss', loss, global_step) # tensorboardX for iter
        writer.add_scalar('total train loss',total_train_loss/len(train_dataloader),epoch) # tensorboardX for epoch
        #---------------------------------------------------------------------------------------------------------
        scheduler.step()
        
        if args.save_checkpoint[-1] == '/':
            args.save_checkpoint = args.save_checkpoint[:-1]
        '''    
        savefilename = args.save_checkpoint+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
                'epoch': epoch,
                'state_dict': first_network.state_dict(),
                'train_loss': total_train_loss/len(train_dataloader),
            }, savefilename)
        '''
        latestfilename =  args.visualize_path + '/checkpoints/checkpoint_latest.tar'  
        torch.save(first_network.state_dict(), latestfilename)
        #-----------------------------------------------------------------------------

        # Valid ----------------------------------------------------------------------------------------------------
        total_val_loss = 0
        total_val_crop_rmse = 0
        print('-------------Validate Epoch', epoch, '-----------')
        first_network.eval()
        for batch_idx, (rgb, depth, depth_mask) in tqdm(enumerate(val_dataloader),desc='it'):

            rgb, depth, depth_mask = rgb.cuda(), depth.cuda(), depth_mask.cuda()

            val_output = val(rgb, depth, depth_mask, batch_idx)
            
            compute_eval_metrics(val_output, depth, depth_mask)
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
        row = [epoch, '{:.4f}'.format(abs_rel_error_meter.avg.item()), 
            '{:.4f}'.format(sq_rel_error_meter.avg.item()), 
            '{:.4f}'.format(torch.sqrt(lin_rms_sq_error_meter.avg).item()),
            '{:.4f}'.format(torch.sqrt(log_rms_sq_error_meter.avg).item()), 
            '{:.4f}'.format(d1_inlier_meter.avg.item()), 
            '{:.4f}'.format(d2_inlier_meter.avg.item()), 
            '{:.4f}'.format(d3_inlier_meter.avg.item())]
        with open(csv_filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerow(row)

    writer.close()
    # End Training
    print("Training Ended hahahaha!!!")
    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
#----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
        