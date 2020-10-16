import cv2
import numpy as np
import argparse
import os
import OpenEXR, Imath, array
import multiprocessing as mp
from cvt import e2p, utils

parser = argparse.ArgumentParser(description='generate_perspective')
parser.add_argument('--row', type=int, default=6, help='num of rows')
parser.add_argument('--col', type=int, default=8, help='num of columns')
parser.add_argument('--hfov', type=int, default=80, help='horizontal fov')
parser.add_argument('--vfov', type=int, default=70, help='vertical fov')
parser.add_argument('--train', action='store_true', default=False,
                    help='train or test')
args = parser.parse_args()
train_file = np.loadtxt('train.txt', dtype=str)
test_file = np.loadtxt('test.txt', dtype=str)
root_path = '../360SD-net/data/Realistic'
train_dir = 'pano_perspective_data/train_stanford'
test_dir = 'pano_perspective_data/test_stanford'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)    

if args.train:
    files = train_file
    dirs = train_dir
    txtfile = open('train_perspective.txt', 'w')
else:
    files = test_file
    dirs = test_dir  
    txtfile = open('test_perspective.txt', 'w')  

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


num_rows = args.row # parameters to control number of views
num_cols = args.col # parameters to control number of views: num_cols * num_rows
rows = np.linspace(-90.0, 90.0, num_rows + 1)
rows = (rows[:-1] + rows[1:]) * 0.5
cols = np.linspace(-180.0, 180.0, num_cols + 1)
cols = (cols[:-1] + cols[1:]) * 0.5
h_fov = args.hfov
v_fov = args.vfov
out_hw = (240, 320)

#for i, filename in enumerate(train_file):
def f(filename):
    if 'area' in filename[0]:
        rgb = cv2.imread(root_path + filename[0])
        depth = read_exr(root_path + filename[3])
        depth = np.expand_dims(depth, -1)
        basename = os.path.splitext(os.path.basename(filename[0]))[0]
        for v in rows:
            for u in cols:
                pers_img = e2p(rgb, (h_fov, v_fov), u, v, out_hw)
                pers_depth = e2p(depth, (h_fov, v_fov), u, v, out_hw)
                pers_depth[pers_depth>8] = 0
                np.savez_compressed("{}/{}_pers_{}_{}.npz".format(dirs, basename, u, v), 
                     depth=pers_depth[:,:,0], color=pers_img)
                txtfile.write("{}/{}_pers_{}_{}.npz\n".format(dirs, basename, u, v))    
    

p = mp.Pool(processes=mp.cpu_count())
p.map(f,files)
p.close()
p.join()   

txtfile.close()