# PanoDepth
![image](https://user-images.githubusercontent.com/13631958/156437825-1ffea1a3-0b08-47d2-8d44-3b0db8d90510.png)
## Getting Started
#### Requirements
- Python (tested on 3.7.4)
- PyTorch (tested on 1.4.0)
- Other dependencies
## Datasets
We train and evaluate on [Stanford2D3D](http://buildingparser.stanford.edu/dataset.html), [360D](https://vcl3d.github.io/3D60/), and [360 stereo dataset](https://vcl3d.github.io/3D60/download.html).

## Usage
### Train one-stage monocular depth estimation, run:
```
python main_mono.py
```
### Train two-stage with the first coarse stage fixed, run:
```
python main_fullpipeline_pretrain.py
```
### Train two-stage end-to-end, run:
```
python main_fullpipeline.py
```  
### Train 360 stereo matching only, if sample on disparity, use:
```
python main_stereo_disp.py
```  
### Train 360 stereo matching only, if sample on depth, use:
```
python main_stereo_depth.py
``` 
### You can change the two-stage configurations in the args. 
### ```--baseline``` defines the novel view synthesis baseline from the input view. ```--nlabels``` defines the number of hypothesis planes for cascade levels. ```--interval``` defines the depth interval for the second cascade level.

### Here are some result comparisons.
![image](https://user-images.githubusercontent.com/13631958/156440224-80427274-0365-4799-a85d-0d5d96bc2875.png)

If you find our code/models useful, please consider citing our paper:
```
@inproceedings{li2021panodepth,
  title={PanoDepth: A Two-Stage Approach for Monocular Omnidirectional Depth Estimation},
  author={Li, Yuyan and Yan, Zhixin and Duan, Ye and Ren, Liu},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  pages={648--658},
  year={2021},
  organization={IEEE}
}
```


