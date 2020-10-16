##to generate train/test files:
1. Define hfov and vfov in generate_pers_img.py, it will generate a total number of rows x cols images.
2. Use --train to generate train files, otherwise test files are generated.

##to train perspective:
1. Use --pretrain to load pretrain weights, learning rate for fine tuning should be set to 1e-5
2. To train from scratch, do not use args input. Learning rate can be set to 1e-4.

