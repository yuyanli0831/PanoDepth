import torch
import torch.utils.data
import numpy as np
from skimage import io
import OpenEXR, Imath, array
import scipy.io
import math
import os.path as osp
from cvt import e2p, utils
import cv2

def uv_meshgrid(w, h):
    uv = np.stack(np.meshgrid(range(w), range(h)), axis=-1)
    uv = uv.astype(np.float64)
    uv[..., 0] = ((uv[..., 0] + 0.5) / w - 0.5) * 2 * np.pi
    uv[..., 1] = ((uv[..., 1] + 0.5) / h - 0.5) * np.pi
    return uv

class OmniDepthDataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for effiicient loading'''

    def __init__(self, 
        root_path, 
        path_to_img_list, 
        image_size,
        ):

        # Set up a reader to load the panos
        self.root_path = root_path

        # Create tuples of inputs/GT
        self.image_list = np.loadtxt(path_to_img_list, dtype=str)

        # Max depth for GT
        self.max_depth = 8.0
        self.min_depth = 0.1
        self.out_hw = image_size


    def __getitem__(self, idx):
        '''Load the data'''

        # Select the panos to load
        relative_paths = self.image_list[idx]
 
        # Load the panos
        relative_basename = osp.splitext((relative_paths))[0]
        basename = osp.splitext(osp.basename(relative_paths))[0]
        mat = np.load(self.root_path + relative_paths)
        pers_rgb = mat['color'] / 255
        pers_depth = mat['depth']
        pers_depth = np.expand_dims(pers_depth, 0)
        pers_mask = ((pers_depth <= self.max_depth) & (pers_depth > self.min_depth)).astype(np.uint8)
 
        # Threshold depths
        pers_depth *= pers_mask
        # Convert to torch format
        pers_rgb = torch.from_numpy(pers_rgb.transpose(2, 0, 1)).float()  #depth
        pers_depth = torch.from_numpy(pers_depth).float()
        pers_mask = torch.from_numpy(pers_mask)
        
        # Return the set of pano data      
        return pers_rgb, pers_depth, pers_mask
        

        
    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.image_list)
