### Welcome to SliceGAN ###
####### Steve Kench #######
'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''

from slicegan import model, networks, util
import argparse
# Define project name
#Project_name = 'COMPOSITE2D'
Project_name = 'compositeSmall2D'
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
#image_type = 'nphases'  # nphases for composite2d.png image
image_type = 'nphases' 
# img_channels should be number of phases for nphase, 3 for colour, or 1 for grayscale
# Even if an image is RGB image, img_channels is not equal to 3. It is the number of different values of Red colour.
img_channels = 256              # 256 for composite2d.png image
#img_channels = 214              
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif2D', 'png', 'jpg', tif3D, 'array')
data_type = 'png'
# Path to your data. One string for isotrpic, 3 for anisotropic
#data_path = ['Examples/compositeSmall2D.png'] # shravan - give 3 paths for 3 scans
data_path = ['Examples/compositeSmall2D.png']

generatePeriodicPrediction = [0, 0, 0]    # shravan - flags to generate periodic microstructures along X, Y and Z directions during the prediction stage
## Network Architectures
# Training image size, no. channels and scale factor vs raw data
img_size, scale_factor = 64,  1  # shravan - img_size is not the size of the input image (input image size is automatically determined by the code). It is the size of the sampled images in a slice (for ex. along X-dimension)
# z vector depth
z_channels = 32     #  These are number of phases used in the one-hot encoded representation of the noise image (used as input to the generator)
# Layers in G and D (Generator and Discriminator)
lays = 5    # shravan - layers for generator
laysd = 6   # shravan - layers for discriminator
dk, gk = [4]*laysd, [4]*lays                                    # kernal sizes [4 4 4 ... laysd times], ...
# gk[0]=8
ds, gs = [2]*laysd, [2]*lays                                    # strides
# gs[0] = 4
df, gf = [img_channels, 64, 128, 256, 512, 1], [    # These also determine the size of neurons in intermediate layers. see output from netG() and netD() below
    z_channels, 1024, 512, 128, 32, img_channels]  # filter sizes for hidden layers (df for discrimination filters and gf for generator filters) --> generator finally outputs an image with same number of channels imag_channels as the reference image.

dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]   # padding for 5 layers of discriminator and generators (Table 1 in the paper)

# other settings
nBatchesBeforeUpdatingGenerator = 8
nSamplesFromRealImages = 32*6  # multiple of 32
ngpu = 1
num_epochs = 60
# batch sizes
batch_size = 4  # shravan - how many samples per batch to load
D_batch_size = 4
# optimiser params for G and D
learningRateGenerator = 0.0001    # <-- for generator
learningRateDiscriminator = 0.0001    # <-- for discriminator
beta1 = 0.9     # <-- same betas are used for both generators and discriminators
beta2 = 0.99
Lambda = 10     # <-- parameter for gradient penalty
latentSpaceSize = 4     # a noise image of size latentSpaceSizeXlatentSpaceSizeXlatentSpaceSize is used as input to Generator to generate images
nWorkers = 0        # number of workers used to load the data

## Create Networks
netD, netG = networks.slicegan_rc_nets(Project_path, Training, image_type, dk, ds, df,dp, gk ,gs, gf, gp) # shravan - retrurns neural networks for discriminator and generator
print('netD: ', netD())

print('netG: ', netG())
# Train
if Training:
    model.train(Project_path, image_type, data_type, data_path, netD, netG, img_channels, img_size, z_channels, scale_factor,nBatchesBeforeUpdatingGenerator,nSamplesFromRealImages,ngpu,num_epochs,batch_size,D_batch_size,learningRateGenerator,learningRateDiscriminator,beta1,beta2,Lambda,latentSpaceSize,nWorkers)
else:
    img, raw, netG = util.test_img(Project_path, image_type, netG(), z_channels, lf=8, periodic=generatePeriodicPrediction)
    
print('The program successfully finished')
