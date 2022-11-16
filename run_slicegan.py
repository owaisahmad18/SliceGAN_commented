### Welcome to SliceGAN ###
####### Steve Kench #######
'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''

from slicegan import model, networks, util
import argparse
# Define project name
Project_name = 'NMC'
# Specify project folder.
Project_dir = 'Trained_Generators'
# Run with False to show an image during or after training
parser = argparse.ArgumentParser()
parser.add_argument('training', type=int)
args = parser.parse_args()
Training = args.training
# Training = 0

Project_path = util.mkdr(Project_name, Project_dir, Training)

## Data Processing
# Define image  type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
image_type = 'colour'
# img_channels should be number of phases for nphase, 3 for colour, or 1 for grayscale
img_channels = 256
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif2D', 'png', 'jpg', tif3D, 'array')
data_type = 'png'
# Path to your data. One string for isotrpic, 3 for anisotropic
data_path = ['Examples/composite2D.png'] # shravan - give 3 paths for 3 scans

## Network Architectures
# Training image size, no. channels and scale factor vs raw data
img_size, scale_factor = 64,  1  # shravan - img_size is not the size of the input image. It is the size of the sliced image
# z vector depth
z_channels = 32
# Layers in G and D (Generator and Discriminator)
lays = 5    # shravan - layers for generator
laysd = 6   # shravan - layers for discriminator
dk, gk = [4]*laysd, [4]*lays                                    # kernal sizes [4 4 4 ... laysd times], ...
# gk[0]=8
ds, gs = [2]*laysd, [2]*lays                                    # strides
# gs[0] = 4
df, gf = [img_channels, 64, 128, 256, 512, 1], [
    z_channels, 1024, 512, 128, 32, img_channels]  # filter sizes for hidden layers (df  for discrimination filters and gf for generator filters)

dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]   # padding for 5 layers of discriminator and generators (Table 1 in the paper)

## Create Networks
netD, netG = networks.slicegan_rc_nets(Project_path, Training, image_type, dk, ds, df,dp, gk ,gs, gf, gp) # shravan - retrurns neural networks for discriminator and generator

# Train
if Training:
    model.train(Project_path, image_type, data_type, data_path, netD, netG, img_channels, img_size, z_channels, scale_factor)
else:
    img, raw, netG = util.test_img(Project_path, image_type, netG(), z_channels, lf=8, periodic=[0, 1, 1])
