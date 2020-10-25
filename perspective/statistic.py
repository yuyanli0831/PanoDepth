import numpy as np
from skimage import io
import Opennpz, Imath, array
import scipy.io
import math
import os.path as osp
from tqdm import tqdm
import cv2

def readDepthPano(path):
    return read_npz(path)[...,0].astype(np.float32)
        #mat_content = np.load(path)
        #depth_img = mat_content['depth']
        #return depth_img.astype(np.float32)

def read_npz(image_fpath):
    f = Opennpz.InputFile( image_fpath )
    dw = f.header()['dataWindow']
    w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)    
    im = np.empty( (h, w, 3) )

    # Read in the npz
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = f.channels( ["R", "G", "B"], FLOAT )
    for i, channel in enumerate( channels ):
        im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
    return im

path_to_img_list = '../360SD-net/data/Realistic'
image_list = np.loadtxt('train.txt', dtype=str)
min_d = np.ones([256, 512], dtype=np.float32) * 1000
max_d = np.zeros([256, 512], dtype=np.float32)
for idx, name in enumerate(tqdm(image_list)):
        # Load the panos
    relative_basename = osp.splitext((name[0]))[0]
    basename = osp.splitext(osp.basename(name[0]))[0]
    depth_down = readDepthPano(path_to_img_list + name[3])
    depth_up = readDepthPano(path_to_img_list + name[5])
    depth_down[depth_down>8] = 0
    depth_up[depth_up>8] = 0
    min_d = np.minimum(depth_down, min_d)
    min_d = np.minimum(depth_up, min_d)
    max_d = np.maximum(depth_down, max_d)
    max_d = np.maximum(depth_up, max_d)

min_d = cv2.normalize(min_d,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
max_d = cv2.normalize(max_d,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow('min', min_d)
cv2.imshow('max', max_d)
cv2.waitKey()
    




