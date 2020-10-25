import cv2
import numpy as np
import argparse
import os
#import OpenEXR, Imath, array
import multiprocessing as mp
from cvt import e2p, utils

parser = argparse.ArgumentParser(description='generate_perspective')
parser.add_argument('--row', type=int, default=4, help='num of rows')
parser.add_argument('--col', type=int, default=6, help='num of columns')
parser.add_argument('--vfov', type=int, default=80, help='vertical fov')
parser.add_argument('--hfov', type=int, default=90, help='horizontal fov')
parser.add_argument('--train', action='store_true', default=False,
                    help='train or test')
parser.add_argument('--train_dir', type=str, default='pano_perspective_data/train',
                    help='train file directory')
parser.add_argument('--test_dir', type=str, default='pano_perspective_data/test',
                    help='test file directory')
args = parser.parse_args()
train_file = np.loadtxt('train.txt', dtype=str)
test_file = np.loadtxt('test.txt', dtype=str)
root_path = '../spherical_mvs/data/Realistic'
train_dir = args.train_dir
test_dir = args.test_dir

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


num_rows = args.row # parameters to control number of views
num_cols = args.col # parameters to control number of views: num_cols * num_rows
rows = np.linspace(-90.0, 90.0, num_rows + 1)
rows = (rows[:-1] + rows[1:]) * 0.5
cols = np.linspace(-180.0, 180.0, num_cols + 1)
cols = (cols[:-1] + cols[1:]) * 0.5
h_fov = args.hfov
v_fov = args.vfov
out_hw = (180, 240)

#for i, filename in enumerate(train_file):
def f(filename):
        rgb = cv2.imread(root_path + filename[0])
        depth = np.load(root_path + filename[3])['depth'].astype(np.float32)
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

