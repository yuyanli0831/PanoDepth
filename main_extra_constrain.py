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
from tqdm import tqdm
from dataset_loader import OmniDepthDataset
import cv2
import spherical as S360
import supervision as L
#from network_syn_mapped import SphericalUnet
from network_extra_constraint import ResNet360
from util import *
#import weight_init
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
parser.add_argument('--epochs', type=int, default=120,
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
parser.add_argument('--visualize_path', default='./visualize_extra_constraints',
                    help='save checkpoint path')                    
parser.add_argument('--tensorboard_path', default='logs',
                    help='tensorboard path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--real', action='store_true', default=False,
                    help='adapt to real world images in both training and validation')
parser.add_argument('--seed', type=int, default=1, metavar='S',
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
lr = 0.0002

#-------------------------------------------------------------------
#data loaders
train_dataset = OmniDepthDataset(
		root_path=input_dir, 
        rotate=False,
        flip=True,
        gamma=True,
		path_to_img_list=train_file_list)

train_dataloader = torch.utils.data.DataLoader(
	dataset=train_dataset,
	batch_size=batch_size,
	shuffle=True,
	num_workers=8,
    pin_memory=True,
	drop_last=True)

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
#first_network = SKNet360(norm_type='batchnorm', activation='relu', aspp=False)
first_network = ResNet360()
#second_network = ResNet360(in_channels=6, out_channels=1)

# parallel on multi gpu
first_network = nn.DataParallel(first_network)
first_network = first_network.cuda()


# Load Checkpoint -------------------------------
start_epoch = 0


print('## Batch size: {}'.format(batch_size))  
print('## learning rate: {}'.format(lr))  
print('## Number of first model parameters: {}'.format(sum([p.data.nelement() for p in first_network.parameters()])))
#print('## Number of second model parameters: {}'.format(sum([p.data.nelement() for p in second_network.parameters()])))
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
def train(rgb, depth, normal, mask, batch_idx):
    #mask = mask>0

    # mask value
    #mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask = mask>0
    mask.detach_()

    optimizer.zero_grad()
    # Loss -------------------------------------------- 
    outputs, outputs_normal = first_network(rgb)
    #input_second = torch.cat([rgb, outputs_normal], 1)

    #outputs = second_network(input_second)
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
    normal_loss = torch.sum((1 - F.cosine_similarity(normal, outputs_normal)) * mask) / mask.sum()
    coords = np.stack(np.meshgrid(range(512), range(256)), -1)
    coords = np.reshape(coords, [-1, 2])
    coords += 1
    uv = coords2uv(coords, 512, 256)      
    xyz = uv2xyz(uv)                 #3d coordinates on a sphere
    xyz = torch.from_numpy(xyz).cuda()
    xyz = xyz.unsqueeze(0).repeat(depth.shape[0], 1, 1)
    reconstruct_xyz = xyz * outputs.view(depth.shape[0], -1, 1)         #reconstructed 3d points based on predicted depth
    gt_xyz = xyz * depth.view(depth.shape[0], -1, 1)   
    outputs_normal_reshape = outputs_normal.view(depth.shape[0], 3, -1).permute(0, 2, 1)
    normal_reshape = normal.view(depth.shape[0], 3, -1).permute(0, 2, 1)
    plane_pred = torch.einsum('bij,bij->bi',outputs_normal_reshape, reconstruct_xyz)
    plane = torch.einsum('bij,bij->bi',normal_reshape, gt_xyz)

    plane_loss = L.direct.calculate_berhu_loss(plane_pred.view(-1, 1, 256, 512), plane.view(-1, 1, 256, 512),                
               mask=mask, weights=attention_weights)

    #print(normal_loss.item(), plane_loss.item(), depth_loss.item(), boundary_loss.item())
    loss = normal_loss + plane_loss + depth_loss   
    #--------------------------------------------------
    rgb = rgb[:,:3,:,:].detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()
    depth_prediction = outputs.detach().cpu().numpy()
    depth_prediction[depth_prediction>8] = 0
    normal = F.normalize(normal, dim=1)
    normal = normal.detach().cpu().numpy()
    outputs_normal = F.normalize(outputs_normal, dim=1)
    normal_prediction = outputs_normal.detach().cpu().numpy()

    if batch_idx % visualize_interval == 0 and batch_idx > 0:
            rgb_img = rgb[0, :, :, :].transpose(1,2,0)
            depth_img = depth[0, 0, :, :]
            depth_pred_img = depth_prediction[0, 0, :, :]
            normal_img = normal[0, :, :, :].transpose(1,2,0)
            normal_img = cv2.normalize(normal_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            normal_pred_img = normal_prediction[0, :, :, :].transpose(1,2,0)
            normal_pred_img = cv2.normalize(normal_pred_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            cv2.imwrite('{}/rgb_{}.png'.format(result_view_dir, batch_idx), rgb_img*255)
            cv2.imwrite('{}/normal_{}.png'.format(result_view_dir, batch_idx), normal_img)
            cv2.imwrite('{}/normal_pred_{}.png'.format(result_view_dir, batch_idx), normal_pred_img)
            plot.imsave('{}/depth_gt_{}.png'.format(result_view_dir, batch_idx), depth_img, cmap="jet")
            plot.imsave('{}/depth_pred_{}.png'.format(result_view_dir, batch_idx), depth_pred_img, cmap="jet")

    return loss

# Valid Function -----------------------
def val(rgb, depth, normal, mask, batch_idx):
    
    mask = mask>0 
    with torch.no_grad():
        outputs, outputs_normal = first_network(rgb)
        #input_second = torch.cat([rgb, outputs_normal], 1)
        #outputs = second_network(input_second)
        
    rgb = rgb[:,:3,:,:].detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()
    depth_prediction = outputs.detach().cpu().numpy()
    depth_prediction[depth_prediction>8] = 0
    normal = F.normalize(normal, dim=1)
    normal = normal.detach().cpu().numpy()
    outputs_normal = F.normalize(outputs_normal, dim=1)
    normal_prediction = outputs_normal.detach().cpu().numpy()

    if batch_idx % 20 == 0:
            rgb_img = rgb[0, :, :, :].transpose(1,2,0)
            depth_img = depth[0, 0, :, :]
            depth_pred_img = depth_prediction[0, 0, :, :]
            normal_img = normal[0, :, :, :].transpose(1,2,0)
            normal_img = cv2.normalize(normal_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            normal_pred_img = normal_prediction[0, :, :, :].transpose(1,2,0)
            normal_pred_img = cv2.normalize(normal_pred_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite('{}/test_normal_{}.png'.format(result_view_dir, batch_idx), normal_img)
            cv2.imwrite('{}/test_normal_pred_{}.png'.format(result_view_dir, batch_idx), normal_pred_img)
            plot.imsave('{}/test_depth_gt_{}.png'.format(result_view_dir, batch_idx), depth_img, cmap="jet")
            plot.imsave('{}/test_depth_pred_{}.png'.format(result_view_dir, batch_idx), depth_pred_img, cmap="jet")
    return outputs

# Adjust Learning Rate
"""
def adjust_learning_rate(optimizer, epoch):
    
    lr = 0.0002
    if epoch > args.start_decay:
        lr *= 0.5
     
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
"""

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
    for epoch in range(start_epoch+1, args.epochs+1):
        print('---------------Train Epoch', epoch, '----------------')
        total_train_loss = 0
        
        #adjust_learning_rate(optimizer, epoch)
        #-------------------------------
        first_network.train()
        # Train --------------------------------------------------------------------------------------------------
        for batch_idx, (rgb, depth, depth_mask) in tqdm(enumerate(train_dataloader),desc='it'):
 
            rgb, depth, depth_mask = rgb.cuda(), depth.cuda(), depth_mask.cuda()
            normal, edge = depth2normal_gpu(depth)
            loss = train(rgb, depth, normal, depth_mask, batch_idx)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            global_step += 1
            if batch_idx % 60 == 0 and batch_idx > 0:
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
        latestfilename =  args.visualize_path + '/checkpoint_latest.tar'  
        torch.save({
                'epoch': epoch,
                'state_dict': first_network.state_dict(),
                #'state_dict2': second_network.state_dict(),
                }, latestfilename)
        #-----------------------------------------------------------------------------

        # Valid ----------------------------------------------------------------------------------------------------
        total_val_loss = 0
        total_val_crop_rmse = 0
        print('-------------Validate Epoch', epoch, '-----------')
        first_network.eval()
        for batch_idx, (rgb, depth, depth_mask) in tqdm(enumerate(val_dataloader),desc='it'):

            rgb, depth, depth_mask = rgb.cuda(), depth.cuda(), depth_mask.cuda()
            normal, edge = depth2normal_gpu(depth)
            val_output = val(rgb, depth, normal, depth_mask, batch_idx)
            
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
        writer.add_scalar('Avg Abs Rel', abs_rel_error_meter.avg, epoch+1)
        writer.add_scalar('Avg Sq Rel', sq_rel_error_meter.avg, epoch+1)
        writer.add_scalar('Avg. Lin. RMS', math.sqrt(lin_rms_sq_error_meter.avg), epoch+1)
        writer.add_scalar('Avg. Log. RMS', math.sqrt(log_rms_sq_error_meter.avg), epoch+1)
        writer.add_scalar('Inlier D1', d1_inlier_meter.avg, epoch+1)
        writer.add_scalar('Inlier D2', d2_inlier_meter.avg, epoch+1)
        writer.add_scalar('Inlier D3', d3_inlier_meter.avg, epoch+1)
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
        