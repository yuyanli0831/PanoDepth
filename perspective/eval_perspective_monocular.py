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
from math import pi
from metrics import *
from torch.utils.tensorboard import SummaryWriter
from skimage.measure import compare_mse as mse
from tqdm import tqdm
from dataset_perspective import OmniDepthDataset
import cv2
import supervision as L
from model_fcrn import FCRN
from util import load_partial_model
#import weight_init
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
from weights import load_weights
from cvt import e2p, utils
import scipy.io
from skimage import io
#from helper import *

parser = argparse.ArgumentParser(description='360SD-Net')
parser.add_argument('--model', default='psmnet',
                    help='select model')
parser.add_argument('--mode', default='pers',
                    help='perspective')
parser.add_argument('--sph_input_dir', default='../spherical_mvs/data/Realistic',
                    help='input data directory')
parser.add_argument('--pers_input_dir', default='./',
                    help='input data directory')
parser.add_argument('--testfile', default='test_perspective.txt',
                    help='validation file name')
parser.add_argument('--batch', type=int, default=1,
                    help='number of batch to inference')
parser.add_argument('--hfov', type=int, default=90,
                    help='horizontal field of view')
parser.add_argument('--vfov', type=int, default=80,
                    help='vertical field of view')
parser.add_argument('--rows', type=int, default=4,
                    help='# of rows')
parser.add_argument('--cols', type=int, default=6,
                    help='# of cols')
parser.add_argument('--checkpoint_dir', default='visualize_80_90_4x6',
                    help='load checkpoint path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Save Checkpoint -------------------------------------------------------------
#if not os.path.isdir(args.save_checkpoint):
#    os.makedirs(args.save_checkpoint)
args.result = os.path.join('..', 'results')
args.type = "pers_laina"
# tensorboard Path -----------------------
#writer_path = os.path.join(args.save_checkpoint,args.tensorboard_path)
#if not os.path.isdir(writer_path):
#    os.makedirs(writer_path)
writer = None #SummaryWriter(writer_path)

#-----------------------------------------

# Random Seed -----------------------------
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#------------------------------------------
#-------------------------------------------------------------------
batch_size = args.batch
max_depth = 8.0
min_depth = 0.1
output_size = (180, 240)
#----------------------------------------------------------
#first network, coarse depth estimation
# option 1, resnet 360 
num_gpu = torch.cuda.device_count()
first_network = FCRN(batch_size=batch_size//num_gpu, output_size=output_size)
#first_network = convert_model(first_network)
# option 2, spherical unet
#first_network = SphericalUnet()

# optian 3, rectnet
#first_network = RectNet()

# parallel on multi gpu
first_network = nn.DataParallel(first_network)
first_network.cuda()
first_network.eval()

#----------------------------------------------------------


# Load Checkpoint -------------------------------
state_dict = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint_latest.tar'))
first_network.load_state_dict(state_dict['state_dict'])
start_epoch = state_dict['epoch']

print('## Batch size: {}'.format(batch_size))  
print('## Number of first model parameters: {}'.format(sum([p.data.nelement() for p in first_network.parameters()])))
#--------------------------------------------------

def cart2sph(x, y, z):
    hxz = np.hypot(x, z)
    theta = np.arctan2(x, z)
    phi = np.arctan2(y, hxz)
    r = np.hypot(hxz, y)
    return theta, phi, r

def pts2sph_with_weight(pts, colors, depth, weight, n):
    theta, phi, r = cart2sph(pts[:,0], pts[:,1], pts[:,2])
    nTheta = n
    nPhi = int(n/2)
    ix = np.floor((theta+math.pi)/(2*math.pi)*nTheta)
    iy = np.floor((phi+math.pi/2)/math.pi*nPhi)
    ix[ix>nTheta] = nTheta
    ix = ix.astype(np.int32)
    iy = iy.astype(np.int32)
    iy[iy>nPhi] = nPhi
    sphimage = np.zeros((nPhi*nTheta, 3), dtype = np.int32)
    sphdepth = np.zeros((nPhi*nTheta), dtype = np.float32)
    sphweight = np.zeros((nPhi*nTheta), dtype = np.float32)
    idx = ix + (iy - 1)*nTheta
    for ch in range(3):
        sphimage[idx, ch] = colors[:, ch]
    sphimage = np.reshape(sphimage, (nPhi, nTheta, 3))  
    sphdepth[idx] = depth
    sphdepth = np.reshape(sphdepth, (nPhi, nTheta))  
    sphweight[idx] = weight
    sphweight = np.reshape(sphweight, (nPhi, nTheta))  

    return sphimage, sphdepth, sphweight

def transform_image_coor2_world(image, depth, K, rot_mat):
    _, channel, row, col = image.shape
    coor_x, coor_y = np.meshgrid(np.arange(0, col), np.arange(0, row))
    image_pos = np.ones((3, row*col), dtype = np.float32)
    image_pos[0, :] = coor_x.flatten(order='F')
    image_pos[1, :] = coor_y.flatten(order='F')
    new_image_pos = np.dot(K, image_pos)

    new_image_pos = np.dot(rot_mat, new_image_pos)

    new_image_pos = np.transpose(new_image_pos)
    transformed_color = np.zeros((row*col, 3), dtype = np.int32)
    transform_depth = np.zeros((row*col,), dtype = np.int32)
    for ch in range(channel):
        transformed_color[:, ch] = image[0, ch, :, :].flatten(order='F')        
    transformed_color = np.flip(transformed_color, 1)
    transform_depth = depth.flatten(order='F')
    x_norm = coor_x.flatten(order='F').astype(np.float32) / col - 0.5
    y_norm = coor_y.flatten(order='F').astype(np.float32) / row - 0.5
    weight = (0.5 - np.abs(x_norm)) * (0.5 - np.abs(y_norm))
    weight = np.maximum(np.zeros_like(weight), weight)
    #new_image_pos[: 2] = 1
    new_image_pos = new_image_pos / np.linalg.norm(new_image_pos, axis=-1)[:, np.newaxis]
    return new_image_pos, transformed_color, transform_depth, weight


def inference(img):
    with torch.no_grad():
        outputs = first_network(img)
    return outputs

def readRGBPano(path):
    '''Read RGB and normalize to [0,1].'''
    #rgb = io.imread(path).astype(np.float32) / 255.
    #rgb = io.imread(path)
    rgb = cv2.imread(path)
    return rgb


def readDepthPano(path):
    #return read_npz(path)[...,0].astype(np.float32)
    mat_content = np.load(path)
    depth_img = mat_content['depth'].astype(np.float32)
    return depth_img

def pers2pano(img, num_rows, num_cols, h_fov, v_fov, out_hw, args, basename):
    rows = np.linspace(-90.0, 90.0, num_rows + 1)
    rows = (rows[:-1] + rows[1:]) * 0.5
    cols = np.linspace(-180.0, 180.0, num_cols + 1)
    cols = (cols[:-1] + cols[1:]) * 0.5
    fx = out_hw[1] * 0.5 / math.tan(h_fov * 0.5 * pi / 180)
    fy = out_hw[0] * 0.5 / math.tan(v_fov * 0.5 * pi / 180)
    K = [[fx, 0.0, out_hw[1] * 0.5],
         [0.0, fy, out_hw[0] * 0.5],
         [0.0, 0.0, 1.0]]
    print ("K: ", K)
    stitched = np.zeros_like(img).astype(np.float32)
    accum_weight = np.zeros((stitched.shape[0], stitched.shape[1])).astype(np.float32)
    accum_weight_3ch = np.zeros_like(stitched).astype(np.float32)
    stitched_depth = np.zeros((stitched.shape[0], stitched.shape[1]), dtype=np.float32)    
    for v in rows:
        for u in cols:
            pers_rgb_np = e2p(img, (h_fov, v_fov), u, v, out_hw)
            if (pers_rgb_np.shape[0] != out_hw[0] or pers_rgb_np.shape[1] != out_hw[1]):
                pers_rgb_np = cv2.resize(pers_rgb_np, (out_hw[1], out_hw[0]))
            pers_rgb = torch.from_numpy(pers_rgb_np.transpose(2, 0, 1)).float().cuda()
            pers_rgb = pers_rgb.unsqueeze(0)
            pers_rgb /= 255
            pers_depth_pred = inference(pers_rgb)
            
            u_rad = u * pi / 180
            v_rad = v * pi / 180
            v_rot = utils.rotation_matrix(v_rad, [1, 0, 0])
            u_rot = utils.rotation_matrix(u_rad, [0, 1, 0])
            rot_mat = u_rot.dot(v_rot)
            transform_coor, transform_color, transform_depth, transform_weight = transform_image_coor2_world(pers_rgb.data.cpu().numpy(), 
                pers_depth_pred.data.cpu().numpy(), np.linalg.inv(K), rot_mat)
            sphimage_i, sphdepth_i, sphweight = pts2sph_with_weight(transform_coor, transform_color, transform_depth, 
                transform_weight, img.shape[1])
            sphweight_3ch = np.concatenate((sphweight[:, :, np.newaxis], sphweight[:, :, np.newaxis], sphweight[:, :, np.newaxis]), 
                axis = 2)
            stitched_depth += sphdepth_i.astype(np.float32) * sphweight #np.maximum(stitched_depth, sphdepth_i.astype(np.float32))        
            #stitched_depth = np.maximum(stitched_depth, sphdepth_i.astype(np.float32))       
            accum_weight += sphweight
            accum_weight_3ch += sphweight_3ch
            #fill_pixels(stitched, pers_img, u, v, h_fov, v_fov, out_hw)
            #quit()
            stitched += sphimage_i * sphweight_3ch
    idx = accum_weight > 0
    stitched[idx] = stitched[idx] / accum_weight_3ch[idx]
    stitched = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR)
    stitched_depth = stitched_depth / accum_weight
    #cv2.imwrite('blended_{}_{}.png'.format(num_rows, num_cols), (stitched * 1.0).astype(np.uint8))
    print (accum_weight.min(), accum_weight.max())
    #norm_weight = accum_weight * 255 #cv2.normalize(accum_weight,  None, 0, 255, cv2.NORM_MINMAX)
    #norm_weight = cv2.applyColorMap(norm_weight.astype(np.uint8), cv2.COLORMAP_JET)
    #cv2.imwrite("norm_weight_{}_{}.png".format(num_rows, num_cols), norm_weight)
    #cv2.imwrite('blended_depth_{}_{}.png'.format(num_rows, num_cols), stitched_depth.astype(np.uint16))
    return (stitched * 1.0).astype(np.uint8), stitched_depth, accum_weight
    
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
    median_scaling_factor = ((gt_depth[depth_mask>0]).median() / depth_pred[depth_mask>0]).median()
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

def get_folder_name(args, h_fov, v_fov, rows, cols):
    current_time = time.strftime('%Y-%m-%d@%H-%M')
    prefix = "fusion_"
    return os.path.join(args.result,
        prefix + 'mode={}.hfov={}.vfov={}.rows={}.cols={}.time={}'.
        format(args.mode, h_fov, v_fov, rows, cols, current_time))


fieldnames = ['imgid','absrel', 'squared_rel', 'alinrmse', 'alogrmse', 'ind1',
                'ind2', 'ind3']

def save_avg_meters(csvfile_name, average_meter, imgid):                
    with open(csvfile_name, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow( {'imgid': imgid,
        				  'absrel': average_meter[0].data.cpu().numpy(),
                  'squared_rel': average_meter[1].data.cpu().numpy(),
                  'alinrmse': average_meter[2],
                  'alogrmse': average_meter[3],
                  'ind1': average_meter[4].data.cpu().numpy(),
                  'ind2': average_meter[5].data.cpu().numpy(),
                  'ind3': average_meter[6].data.cpu().numpy()})
# Main Function ---------------------------------------------------------------------------------------------
def main():
    global_step = 0
    global_val = 0
    image_list = np.loadtxt('test.txt', dtype=str)
    out_hw = (180, 240)
    h_fov = args.hfov #136
    v_fov = args.vfov #120
    rows = args.rows
    cols = args.cols
    result_dir = 'result_visualize'

    if (not os.path.exists(result_dir)):
        os.makedirs(result_dir)
    #csvfile_name = os.path.join(result_dir, "error.csv")
    #with open(csvfile_name, 'w') as csvfile:
    #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #    writer.writeheader()
    imgid = 0
    for rel_path in image_list: 
        relative_name = os.path.splitext((rel_path[0]))[0]
        basename = os.path.basename(relative_name)
        print (relative_name, basename)
        sph_img_np = readRGBPano(args.sph_input_dir + rel_path[0])
        sph_depth_np = readDepthPano(args.sph_input_dir + rel_path[3])
        sph_rgb = torch.from_numpy(sph_img_np.transpose(2, 0, 1)).float()

        # get perspective views
        stitched_rgb, stitched_depth, accum_weight = pers2pano(sph_img_np, rows, cols, h_fov, v_fov, out_hw, args, basename)
        stitched_depth[np.isnan(stitched_depth)] = 0
        
        sph_mask_np = ((sph_depth_np <= max_depth) & (sph_depth_np > min_depth) & (stitched_depth <= max_depth) & (stitched_depth > min_depth)).astype(np.uint8)
        sph_depth_np *= sph_mask_np
        #stitched_depth *= sph_mask_np
        #sph_mask_np = ((sph_depth_np <= max_depth) & (sph_depth_np > min_depth)).astype(np.uint8)
        sph_depth_torch = torch.from_numpy(sph_depth_np).float()
        stitched_depth_torch = torch.from_numpy(stitched_depth).float()
        sph_mask_torch = torch.from_numpy(sph_mask_np).float()
        compute_eval_metrics(sph_depth_torch, stitched_depth_torch, sph_mask_torch)
        #depth_residual = np.abs(sph_depth_np - stitched_depth) * sph_mask_np
        #depth_residual_display = cv2.applyColorMap((depth_residual / 8.0 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #cv2.imwrite(os.path.join(result_dir, "residual_{}.png".format(imgid)), depth_residual_display)
        #cv2.imwrite(result_dir + '/down_' + str(imgid) + '.png', sph_img_np)
        #plot.imsave(result_dir + '/depth_gt_' + str(imgid) + '.png', sph_depth_np, cmap="viridis")
        plot.imsave(result_dir + '/depth_pred_' + str(imgid) + '.png', stitched_depth, cmap="viridis")
        avg_meter = [abs_rel_error_meter.avg,
                        sq_rel_error_meter.avg, 
                        math.sqrt(lin_rms_sq_error_meter.avg),
                        math.sqrt(log_rms_sq_error_meter.avg),
                        d1_inlier_meter.avg,
                        d2_inlier_meter.avg,
                        d3_inlier_meter.avg]
        #save_avg_meters(csvfile_name, avg_meter, imgid)
        #print ("[{}]: {}, {}, {}, {}, {}, {}".format(rel_path, depth_residual.min(), depth_residual.max(), sph_depth_np.min(), sph_depth_np.max(), stitched_depth.min(), stitched_depth.max()))
        imgid += 1        
        #if (imgid > 10):
        #    break

    print('# of images: {}\n'
        '  Avg. Abs. Rel. Error: {:.4f}\n'
        '  Avg. Sq. Rel. Error: {:.4f}\n'
        '  Avg. Lin. RMS Error: {:.4f}\n'
        '  Avg. Log RMS Error: {:.4f}\n'
        '  Inlier D1: {:.4f}\n'
        '  Inlier D2: {:.4f}\n'
        '  Inlier D3: {:.4f}\n\n'.format(
        imgid, 
        abs_rel_error_meter.avg,
        sq_rel_error_meter.avg,
        math.sqrt(lin_rms_sq_error_meter.avg),
        math.sqrt(log_rms_sq_error_meter.avg),
        d1_inlier_meter.avg,
        d2_inlier_meter.avg,
        d3_inlier_meter.avg))

    # End Training
    
    print("Inference Ended hahahaha!!!")
#----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
        
