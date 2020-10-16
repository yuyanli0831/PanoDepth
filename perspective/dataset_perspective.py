import torch
import torch.utils.data
import numpy as np
from skimage import io
import OpenEXR, Imath, array
import scipy.io
import math
import os.path as osp
from cvt import e2p, utils

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
        rotate=False,
        flip=False,
        gamma=False):

        # Set up a reader to load the panos
        self.root_path = root_path

        # Create tuples of inputs/GT
        self.image_list = np.loadtxt(path_to_img_list, dtype=str)

        # Max depth for GT
        self.max_depth = 8.0
        self.min_depth = 0.1
        self.rotate = rotate
        self.flip = flip
        self.gamma = gamma
        self.num_rows = 3
        self.num_cols = 4
        self.h_fov = 136 
        self.v_fov = 120
        self.out_hw = (240, 320)


    def __getitem__(self, idx):
        '''Load the data'''

        # Select the panos to load
        relative_paths = self.image_list[idx]

        data = np.load(self.root_path + relative_paths)
        down = data['color']
        depth_down = data['depth']
        # Load the panos
        #relative_basename = osp.splitext((relative_paths[0]))[0]
        #basename = osp.splitext(osp.basename(relative_paths[0]))[0]
        #down = self.readRGBPano(self.root_path + relative_paths[0])
        #depth_down = self.readDepthPano(self.root_path + relative_paths[3])
      
        down = down.astype(np.float32)/255
        # Random flip
        if self.flip:
          if np.random.randint(2) == 0:
            down = np.flip(down, axis=1)
            depth_down = np.flip(depth_down, axis=1)

        # Random horizontal rotate
        if self.rotate:
          dx = np.random.randint(down.shape[1])
          down = np.roll(down, dx, axis=1)
          depth_down = np.roll(depth_down, dx, axis=1)

        # Random gamma augmentation
        if self.gamma:
          p = np.random.uniform(1, 2)
          if np.random.randint(2) == 0:
            p = 1 / p
            down = down ** p

        depth_down = np.expand_dims(depth_down, 0)
        mask = ((depth_down <= self.max_depth) & (depth_down > self.min_depth)).astype(np.uint8)
        rgb = torch.from_numpy(down.transpose(2, 0, 1)).float()
        depth = torch.from_numpy(depth_down).float()
        mask = torch.from_numpy(mask)
        return rgb, depth, mask
        '''
        rows = np.linspace(-90.0, 90.0, self.num_rows + 1)
        rows = (rows[:-1] + rows[1:]) * 0.5
        cols = np.linspace(-180.0, 180.0, self.num_cols + 1)
        cols = (cols[:-1] + cols[1:]) * 0.5
        depth_down = np.expand_dims(depth_down, -1)
        pers_rgb, pers_depth = [], []
        for v in rows:
            for u in cols:
               pers_rgb.append(e2p(down, (self.h_fov, self.v_fov), u, v, self.out_hw))
               pers_depth.append(e2p(depth_down, (self.h_fov, self.v_fov), u, v, self.out_hw))

        pers_rgb = np.stack(pers_rgb, 0)
        pers_depth = np.stack(pers_depth, 0)
        
        pers_mask = ((pers_depth <= self.max_depth) & (pers_depth > self.min_depth)).astype(np.uint8)

        # Threshold depths
        pers_depth *= pers_mask

        # Convert to torch format
        pers_rgb = torch.from_numpy(pers_rgb.transpose(0,3,1,2)).float()  #depth
        pers_depth = torch.from_numpy(pers_depth.transpose(0,3,1,2)).float()
        pers_mask = torch.from_numpy(pers_mask.transpose(0,3,1,2))
        
        # Return the set of pano data
        
        return pers_rgb, pers_depth, pers_mask
        '''

        
    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.image_list)

    def readRGBPano(self, path):
        '''Read RGB and normalize to [0,1].'''
        #rgb = io.imread(path).astype(np.float32) / 255.
        rgb = io.imread(path)
        return rgb


    def readDepthPano(self, path):
        return self.read_exr(path)[...,0].astype(np.float32)
        #mat_content = np.load(path)
        #depth_img = mat_content['depth']
        #return depth_img.astype(np.float32)

    def read_exr(self, image_fpath):
        f = OpenEXR.InputFile( image_fpath )
        dw = f.header()['dataWindow']
        w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)    
        im = np.empty( (h, w, 3) )

        # Read in the EXR
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = f.channels( ["R", "G", "B"], FLOAT )
        for i, channel in enumerate( channels ):
            im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
        return im