import OpenEXR, array, Imath
import glob
import os
import numpy as np
import multiprocessing as mp

files = np.loadtxt('test.txt', dtype='str')

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

def f(fn):
    for i in range(3, 6):
        depth = read_exr('../data/Realistic' + fn[i]) 
        normal = read_exr('../data/Realistic' + fn[i+3]) 
        filename = fn[i].replace('exr', 'npz')   
        np.savez_compressed('../data/Realistic' + filename, depth=depth, normal=normal)

p = mp.Pool(processes=mp.cpu_count())
p.map(f,files)
p.close()
p.join()           