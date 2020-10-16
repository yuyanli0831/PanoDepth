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
from dataset_perspective import OmniDepthDataset
import cv2
import supervision as L
from model_fcrn import FCRN
from util import load_partial_model
import weight_init
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
from weights import load_weights
from cvt import e2p, utils
import scipy.io


parser = argparse.ArgumentParser(description='360SD-Net')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='psmnet',
                    help='select model')
parser.add_argument('--input_dir', default='./',
                    help='input data directory')
parser.add_argument('--trainfile', default='train_perspective.txt',
                    help='train file name')
parser.add_argument('--testfile', default='test_perspective.txt',
                    help='validation file name')
parser.add_argument('--epochs', type=int, default=60,
                    help='number of epochs to train')
parser.add_argument('--start_decay', type=int, default=30,
                    help='number of epoch for lr to start decay')
parser.add_argument('--start_learn', type=int, default=10,
                    help='number of epoch for LCV to start learn')
parser.add_argument('--batch', type=int, default=96,
                    help='number of batch to train')
parser.add_argument('--visualize_interval', type=int, default=60,
                    help='number of batch to train')
parser.add_argument('--baseline', type=float, default=0.24,
                    help='image pair baseline distance')
parser.add_argument('--checkpoint', default= None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='./checkpoints_monocular_depth',
                    help='save checkpoint path')
parser.add_argument('--visualize_path', default='./visualize_monocular_perspective_notransfer',
                    help='save checkpoint path')                    
parser.add_argument('--tensorboard_path', default='logs',
                    help='tensorboard path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pretrain', action='store_true', default=False,
                    help='if load pretrain weights')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Save Checkpoint -------------------------------------------------------------
if not os.path.isdir(args.save_checkpoint):
    os.makedirs(args.save_checkpoint)
# tensorboard Path -----------------------
writer_path = os.path.join(args.save_checkpoint,args.tensorboard_path)
if not os.path.isdir(writer_path):
    os.makedirs(writer_path)
writer = SummaryWriter(writer_path)

#-----------------------------------------
pretrain = args.pretrain
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
baseline = args.baseline
maxdisp = args.maxdisp
visualize_interval = args.visualize_interval
init_lr = 1e-4
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
	num_workers=8,
	drop_last=True)

#----------------------------------------------------------
#first network, coarse depth estimation
# option 1, resnet 360 
num_gpu = torch.cuda.device_count()
first_network = FCRN(batch_size=batch_size//num_gpu, output_size=(240, 320))
if pretrain:
    first_network.load_state_dict(load_weights(first_network, weights_file="NYU_ResNet-UpProj.npy", 
                     dtype=torch.cuda.FloatTensor))
first_network = convert_model(first_network)
# option 2, spherical unet
#first_network = SphericalUnet()

# optian 3, rectnet
#first_network = RectNet()

# parallel on multi gpu
first_network = nn.DataParallel(first_network)
first_network.cuda()
#----------------------------------------------------------


# Load Checkpoint -------------------------------
start_epoch = 0
if args.checkpoint is not None:
    state_dict = torch.load(args.checkpoint)
    first_network.load_state_dict(state_dict['state_dict'])
    start_epoch = state_dict['epoch']

print('## Batch size: {}'.format(batch_size))  
print('## learning rate: {}'.format(init_lr))  
print('## Number of first model parameters: {}'.format(sum([p.data.nelement() for p in first_network.parameters()])))
#--------------------------------------------------

# Optimizer ----------
#optimizer = optim.Adam(list(model.parameters())+list(view_syn_network.parameters()), 
#        lr=0.001, betas=(0.9, 0.999))
optimizer = optim.Adam(list(first_network.parameters()), 
        lr=init_lr, betas=(0.9, 0.999))
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
 
def stitch2pano(pers, output_size=(256,512,3)):
    num_rows = 3 # parameters to control number of views
    num_cols = 4 # parameters to control number of views: num_cols * num_rows
    rows = np.linspace(-90.0, 90.0, num_rows + 1)
    rows = (rows[:-1] + rows[1:]) * 0.5
    cols = np.linspace(-180.0, 180.0, num_cols + 1)
    cols = (cols[:-1] + cols[1:]) * 0.5
    h_fov = 136 
    v_fov = 120
    out_hw = (240, 320)
    fx = out_hw[1] * 0.5 / math.tan(h_fov * 0.5 * math.pi / 180)
    fy = out_hw[0] * 0.5 / math.tan(v_fov * 0.5 * math.pi / 180)
    K = [[fx, 0.0, out_hw[1] * 0.5],
         [0.0, fy, out_hw[0] * 0.5],
         [0.0, 0.0, 1.0]]
    stitched = np.zeros(output_size).astype(np.float32)
    accum_weight = np.zeros((stitched.shape[0], stitched.shape[1])).astype(np.float32)
    accum_weight_3ch = np.zeros_like(stitched).astype(np.float32)
    count = 0
    for v in rows:
        for u in cols:
            pers_img = pers[count, :, :, :]
            u_rad = u * math.pi / 180
            v_rad = v * math.pi / 180
            v_rot = utils.rotation_matrix(v_rad, [1, 0, 0])
            u_rot = utils.rotation_matrix(u_rad, [0, 1, 0])
            rot_mat = u_rot.dot(v_rot)
            transform_coor, transform_color, transform_weight = utils.transform_image_coor2_world(pers_img, 
                np.linalg.inv(K), rot_mat)
            sphimage_i, sphweight = utils.pts2sph_with_weight(transform_coor, transform_color, 
                transform_weight, output_size[1])
            sphweight_3ch = np.concatenate((sphweight[:, :, np.newaxis], sphweight[:, :, np.newaxis], sphweight[:, :, np.newaxis]), axis = 2)
            accum_weight += sphweight
            accum_weight_3ch += sphweight_3ch
            #fill_pixels(stitched, pers_img, u, v, h_fov, v_fov, out_hw)
            #quit()
            stitched += sphimage_i * sphweight_3ch
            count += 1
    idx = accum_weight > 0
    stitched[idx] = stitched[idx] / accum_weight_3ch[idx]

    #stitched = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR)
    return stitched

# Train Function -------------------
def train(img, depth, mask, batch_idx):
    #mask = mask>0

    # mask value
    #mask = (disp_true < args.maxdisp) & (disp_true > 0)

    optimizer.zero_grad()
    # Loss -------------------------------------------- 
    img[torch.isnan(img)] = 0
    depth[torch.isnan(depth)] = 0
    outputs = first_network(img)
    attention_weights = torch.ones_like(depth)

    # berhu depth loss with coordinates attention weight 
    depth_loss = L.direct.calculate_l1_loss(outputs, depth,                
               mask=mask)

    loss = depth_loss        
    #--------------------------------------------------
    gt = img.detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()
    depth_prediction = outputs.detach().cpu().numpy()

    if batch_idx % visualize_interval == 0:
        for i in range(6):
            gt_img = gt[i, :, :, :].transpose(1,2,0)
            depth_down_img = depth[i, 0, :, :]
            depth_down_img /= np.max(depth_down_img)
            #depth_down_img = cv2.normalize(depth_down_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            depth_pred_img = depth_prediction[i, 0, :, :]
            depth_pred_img /= np.max(depth_pred_img)
            #depth_pred_img = cv2.normalize(depth_pred_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite(result_view_dir + '/down_' + str(i) + '.png', gt_img*255)
            plot.imsave(result_view_dir + '/depth_gt_' + str(i) + '.png', depth_down_img, cmap="viridis")
            plot.imsave(result_view_dir + '/depth_pred_' + str(i) + '.png', depth_pred_img, cmap="viridis")
            #cv2.imwrite(result_view_dir + '/depth_gt_' + str(i) + '.png', depth_down_img)
            #cv2.imwrite(result_view_dir + '/depth_pred_' + str(i) + '.png', depth_pred_img)

    return loss

# Valid Function -----------------------
def val(img, depth, mask, batch_idx):
    
    mask = mask>0
    
    with torch.no_grad():
        outputs = first_network(img)
        
    gt = img.detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()
    depth_prediction = outputs.detach().cpu().numpy()
    batch_n, _, h, w = gt.shape

    if batch_idx % visualize_interval == 0:
        for i in range(6):
            gt_img = gt[i, :, :, :].transpose(1,2,0)
            depth_down_img = depth[i, 0, :, :]
            #depth_down_img /= np.max(depth_down_img)
            #depth_down_img = cv2.normalize(depth_down_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            depth_pred_img = depth_prediction[i, 0, :, :]
            #depth_pred_img /= np.max(depth_pred_img)
            #depth_pred_img = cv2.normalize(depth_pred_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite(result_view_dir + '/test_down_' + str(i) + '.png', gt_img*255)
            plot.imsave(result_view_dir + '/test_depth_gt_' + str(i) + '.png', depth_down_img, cmap="viridis")
            plot.imsave(result_view_dir + '/test_depth_pred_' + str(i) + '.png', depth_pred_img, cmap="viridis")
            #cv2.imwrite(result_view_dir + '/depth_gt_' + str(i) + '.png', depth_down_img)
            #cv2.imwrite(result_view_dir + '/depth_pred_' + str(i) + '.png', depth_pred_img)
            
    return outputs

# Adjust Learning Rate
def adjust_learning_rate(optimizer, epoch):
    
    lr = init_lr
    if epoch > args.start_decay:
        lr *= 0.5
     
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
    for epoch in range(start_epoch+1, args.epochs+1):
        print('---------------Train Epoch', epoch, '----------------')
        total_train_loss = 0
        
        adjust_learning_rate(optimizer, epoch)

        #-------------------------------
        first_network.train()
        # Train --------------------------------------------------------------------------------------------------
        for batch_idx, (pers_rgb, pers_depth, pers_mask) in enumerate(train_dataloader):
            #batch_size, n, _, h, w = pers_rgb.shape
            pers_rgb, pers_depth, pers_mask = pers_rgb.cuda(), pers_depth.cuda(), pers_mask.cuda()

            '''    
            pers_rgb = pers_rgb.view(batch_size*n, -1, h, w)
            pers_depth = pers_depth.view(batch_size*n, -1, h, w)
            pers_mask = pers_mask.view(batch_size*n, -1, h, w)
            b = np.arange(batch_size*n)
            np.random.shuffle(b)
            pers_rgb = pers_rgb[b, :, :, :]
            pers_depth = pers_depth[b, :, :, :]
            pers_mask = pers_mask[b, :, :, :]
            '''
            loss = train(pers_rgb, pers_depth, pers_mask, batch_idx)
            
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

        
        if args.save_checkpoint[-1] == '/':
            args.save_checkpoint = args.save_checkpoint[:-1]
        savefilename = args.save_checkpoint+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
                'epoch': epoch,
                'state_dict': first_network.state_dict(),
                'train_loss': total_train_loss/len(train_dataloader),
            }, savefilename)
        #-----------------------------------------------------------------------------

        # Valid ----------------------------------------------------------------------------------------------------
        total_val_loss = 0
        total_val_crop_rmse = 0
        print('-------------Validate Epoch', epoch, '-----------')
        first_network.eval()
        for batch_idx, (pers_rgb, pers_depth, pers_mask) in enumerate(tqdm(val_dataloader)):

            pers_rgb, pers_depth, pers_mask = pers_rgb.cuda(), pers_depth.cuda(), pers_mask.cuda()
            '''
            pers_rgb = pers_rgb.view(batch_size*n, -1, h, w)
            pers_depth = pers_depth.view(batch_size*n, -1, h, w)
            pers_mask = pers_mask.view(batch_size*n, -1, h, w)
            '''
            #val_gt, val_pred = val(pers_rgb, pers_depth, pers_mask, batch_idx)
            #mask = val_gt > 0
            #compute_eval_metrics(val_pred, val_gt, mask)
            val_output = val(pers_rgb, pers_depth, pers_mask, batch_idx)
            compute_eval_metrics(val_output, pers_depth, pers_mask)
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
        
