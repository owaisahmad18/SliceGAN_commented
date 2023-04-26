### Welcome to SliceGAN ###
####### Steve Kench #######
'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''

from slicegan import model, networks, util
import argparse
# Specify project folder.
Project_dir = 'Trained_Generators/WC-720-51-final-cropped/epoch_3_smallerMS'
# Define project name
#Project_name = 'COMPOSITE2D'
Project_name = 'WC-720-51-final-cropped'
# Run with False to show an image during or after training
parser = argparse.ArgumentParser()
parser.add_argument('training', type=int)
args = parser.parse_args()
Training = args.training

size_pixel_real_img = 1.88679245283 # 53pixels = 100microns for the WC-720-51-final-cropped microstructure
if Training ==0:
    print("NOTE: The size of the pixel in the reference image is set to:", size_pixel_real_img)
# Training = 0

Project_path = util.mkdr(Project_name, Project_dir, Training)
filename_prediction_gen_params = Project_path + '_Gen_epoch_3.pt'
filepath_log = Project_path + '.log'
## Data Processing
# Define image  type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
#image_type = 'nphases'  # nphases for composite2d.png image
image_type = 'nphases' 
# img_channels should be number of phases for nphase, 3 for colour, or 1 for grayscale
# Even if an image is RGB image, img_channels is not equal to 3. It is the number of different values of Red colour.
img_channels = 2              # 256 for composite2d.png image, 107 for TC image
#img_channels = 214              
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif2D', 'png', 'jpg', tif3D, 'array')
data_type = 'png'
# Path to your data. One string for isotrpic, 3 for anisotropic
#data_path = ['Examples/compositeSmall2D.png'] # shravan - give 3 paths for 3 scans
data_path = ['Examples/WC-720-51-final-cropped.png']
  
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
nBatchesBeforeUpdatingGenerator = 5
nSamplesFromRealImages = 32*600  # multiple of 32
ngpu = 1
num_epochs = 50000
# batch sizes
batch_size = 8  # shravan - how many samples per batch to load
D_batch_size = 8
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
    model.train(Project_path, filepath_log, image_type, data_type, data_path, netD, netG, img_channels, img_size, z_channels, scale_factor,nBatchesBeforeUpdatingGenerator,nSamplesFromRealImages,ngpu,num_epochs,batch_size,D_batch_size,learningRateGenerator,learningRateDiscriminator,beta1,beta2,Lambda,latentSpaceSize,nWorkers)
else:
    img, raw, netG = util.test_img(Project_path, filename_prediction_gen_params, image_type, data_path[0], size_pixel_real_img, netG(), z_channels, lf=4, periodic=generatePeriodicPrediction)
    
print('The program successfully finished')
