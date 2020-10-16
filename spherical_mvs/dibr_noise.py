import numpy as np
import cv2
from supervision.splatting import render
import torch
import torch.nn as nn
import OpenEXR, Imath, array
import spherical as S360

def read_exr(image_fpath):
        f = OpenEXR.InputFile( image_fpath )
        dw = f.header()['dataWindow']
        w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)    
        im = np.empty( (h, w, 3) )

        # Read in the EXR
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = f.channels( ["R", "G", "B"], FLOAT )
        for i, channel in enumerate( channels ):
            im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
        return im[:, :, 0]

image = cv2.imread('./0_area_31_color_0_Left_Down_0.0.png', -1)
depth = read_exr('./0_area_31_depth_0_Left_Down_0.0.exr')
depth = depth.astype(np.float32)
h, w = depth.shape
depth[depth>8] = 0
gaussian_noise = np.random.normal(loc=0, scale=0.2, size=depth.shape).astype(np.float32)
depth_noise1 = np.clip((depth + gaussian_noise),0,8)
#depth_noise1 = cv2.medianBlur(depth,5)
image = image.astype(np.float32)
image = image.transpose(2, 0, 1)
image = np.expand_dims(image, 0)
image = torch.from_numpy(image)
depth = np.reshape(depth, [1, 1, h, w])
depth = torch.from_numpy(depth)
depth_noise1 = np.reshape(depth_noise1, [1, 1, h, w])
depth_noise1 = torch.from_numpy(depth_noise1)

sgrid = S360.grid.create_spherical_grid(512)
uvgrid = S360.grid.create_image_grid(512, 256)

def dibr(image, depth, sgrid, uvgrid):
    disp = torch.cat(
                (
                    torch.zeros_like(depth),
                    S360.derivatives.dtheta_vertical(sgrid, depth, 0.26)
                ),
                dim=1
            )
    render_coords = uvgrid + disp
    render_coords[torch.isnan(render_coords)] = 0
    render_coords[torch.isinf(render_coords)] = 0
    rendered,_ = render(image, depth, render_coords, max_depth=8)
    rendered = rendered[0,:,:].cpu().numpy()
    rendered = rendered.transpose(1, 2, 0)
    return rendered

render_gt = dibr(image, depth, sgrid, uvgrid)  
render_noise = dibr(image, depth_noise1, sgrid, uvgrid)  
up = cv2.imread('./0_area_31_color_0_Up_0.0.png', -1).astype(np.float32)

def mse(gt, render):
    mask = render>0
    return (((render[mask>0] - gt[mask>0]) ** 2) / gt[mask>0]).mean()

print(mse(up, render_gt))
print(mse(up, render_noise))

cv2.imshow('gt', up.astype(np.uint8))
cv2.imshow('render_gt', render_gt.astype(np.uint8))
cv2.imshow('render_noise', render_noise.astype(np.uint8))
cv2.waitKey()



