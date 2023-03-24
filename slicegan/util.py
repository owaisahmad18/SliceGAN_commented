import os
from torch import nn
import torch
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
## Training Utils

def mkdr(proj,proj_dir,Training):
    """
    When training, creates a new project directory or overwrites an existing directory according to user input. When testing, returns the full project path
    :param proj: project name
    :param proj_dir: project directory
    :param Training: whether new training run or testing image
    :return: full project path
    """    
    pth = proj_dir + '/' + proj
    if Training:
        try:
            os.mkdir(proj_dir)
            os.mkdir(pth)
            return pth + '/' + proj
        except FileExistsError:
            print('Directory', pth, 'already exists. Enter new project name or hit enter to overwrite')
            new = input()
            if new == '':
                return pth + '/' + proj
            else:
                pth = mkdr(new, proj_dir, Training)
                return pth
        except FileNotFoundError:
            print('The specifified project directory ' + proj_dir + ' does not exist. Please change to a directory that does exist and again')
            sys.exit()
    else:
        return pth + '/' + proj


def weights_init(m):
    """
    Initialises training weights
    :param m: Convolution to be intialised
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda,nc):
    """
    calculate gradient penalty for a batch of real and fake data
    :param netD: Discriminator network
    :param real_data:
    :param fake_data:
    :param batch_size:
    :param l: image size
    :param device:
    :param gp_lambda: learning parameter for GP
    :param nc: channels
    :return: gradient penalty
    """
    #sample and reshape random numbers
    alpha = torch.rand(batch_size, 1, device = device)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)

    # create interpolate dataset
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    #pass interpolates through netD
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device = device),
                              create_graph=True, only_inputs=True)[0]
    # extract the grads and calculate gp
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def calc_eta(steps, time, start, i, epoch, num_epochs,isTrainedGeneratorThisBatch):
    """
    Estimates the time remaining based on the elapsed time and epochs
    :param steps:
    :param time: current time
    :param start: start time
    :param i: iteration through this epoch
    :param epoch:
    :param num_epochs: totale no. of epochs
    """
    elap = time - start
    progress = epoch * steps + i    # total number of batches processed so far in all the epochs
    rem = num_epochs * steps - progress # remaining number of batches to process = total number of batches in all the epochs - number of batches processed so far
    ETA = rem / progress * elap   # linearly extrapolate to compute the time remaining (based on number of batches remaining) based on time elapsed (based on batches processed so far)
    hrs = int(ETA / 3600)
    mins = int((ETA / 3600 % 1) * 60)
    print('Epochs: %d/%d \t batchNumber: %d/%d \t Est. Time Remaining: %d hrs %d mins \t isTrainedGeneratorThisBatch: '
          % (epoch+1, num_epochs, i, steps,
             hrs, mins),isTrainedGeneratorThisBatch)

## Plotting Utils
def post_proc(img,imtype):
    """
    turns one hot image back into grayscale
    :param img: input image
    :param imtype: image type
    :return: plottable image in the same form as the training data
    """
    try:
        #make sure it's one the cpu and detached from grads for plotting purposes
        img = img.detach().cpu()
    except:
        pass
    if imtype == 'colour':
        print('img[0]:',img[0])
        return np.int_(255 * (np.swapaxes(img[0], 0, -1)))
    if imtype == 'grayscale':
        return 255*img[0][0]
    else:
        nphase = img.shape[1]
        return 255*torch.argmax(img, 1)/(nphase-1)  # probabilistic interpretation i.e. phase with higher one-hot encoding is more likely to be present in this pixel. see sliceGAN paper.
                                                    # argmax(img, 1) --> gives index of maximum value along dimension 1 for each pixel in img. dimension 1 represents the one-hot-encoded representation of the image.
                                                    # take each of the voxel and go to its one-hot encoded representation (i.e. nphase number of values) and pick the maximum of these. After that, scale it to a gray-scale color map by multiplying with (256-1)/(nphase-1)
def test_plotter(img,slcs,imtype,pth):
    """
    creates a fig with 3*slc subplots showing example slices along the three axes
    :param img: raw input image
    :param slcs: number of slices to take in each dir
    :param imtype: image type
    :param pth: where to save plot
    """
    # img --> one-hot encoded representation of 3D cube (typically 1x256x64x64x64 size i.e. 64x64x64 cube with nphases (==256 here) for which one-hot encoding is done)
    # At this stage, an n-phase image has a size of 1XnPhasesXimg_size(l)Ximg_size(l)Ximg_size(l) i.e. post_proc(img,imtype) has this size
    # post_proc --> turns on-hot-encoding to gray-scale image
    img = post_proc(img,imtype)[0]  # shravan - post-process the image <--- for a colour image type, swaps the 2nd and 4th index. i.e. the img becomes 64x64x64x256 size
    # shravan - at this stage, the img becomes img_size(l)Ximg_size(l)Ximg_size(l) size
    fig, axs = plt.subplots(slcs, 3)    # plots the graphs in 5 rows by 3 columns format if slcs=5. ax returns an array of axes
    if imtype == 'colour':
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :, :], vmin = 0, vmax = 255) # shravan - [j,0]--> jth row and 0th column 
            axs[j, 1].imshow(img[:, j, :, :],  vmin = 0, vmax = 255)    # imshow(data_of_the_image,cmap,vmin=colorbar_min_range,vmax=colorbar_max_range... etc.)
            axs[j, 2].imshow(img[:, :, j, :],  vmin = 0, vmax = 255)
    elif imtype == 'grayscale':
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :], cmap = 'gray')
            axs[j, 1].imshow(img[:, j, :], cmap = 'gray')
            axs[j, 2].imshow(img[:, :, j], cmap = 'gray')
    else:
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :])  # shravan - by default, the 'viridis' color map is used
            axs[j, 1].imshow(img[:, j, :])
            axs[j, 2].imshow(img[:, :, j])
    plt.savefig(pth + '_slices.png')
    plt.close()

def graph_plot(data,labels,pth,name,xlabel,ylabel):
    """
    simple plotter for all the different graphs
    :param data: a list of data arrays
    :param labels: a list of plot labels
    :param pth: where to save plots
    :param name: name of the plot figure
    :return:
    """
    
    for datum,lbl in zip(data,labels):
        #print('datum: ',datum)
        plt.plot(datum, label = lbl)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.subplots_adjust(left=0.15)
    plt.savefig(pth + '_' + name)
    plt.close()


def test_img(pth, imtype, netG, nz = 64, lf = 4, periodic=False):
    """
    saves a test volume for a trained or in progress of training generator
    :param pth: where to save image and also where to find the generator
    :param imtype: image type
    :param netG: Loaded generator class
    :param nz: latent z dimension
    :param lf: length factor
    :param show:
    :param periodic: list of periodicity in axis 1 through n
    :return:
    """
    colormap_display = "viridis"    # colormap for the prediction microstructure. "viridis", "greys", "gray", "binary", "jet", "hsv" etc. similarly, "viridis_r" for reversing the color schemes. see https://matplotlib.org/stable/gallery/color/colormap_reference.html
    netG.load_state_dict(torch.load(pth + '_Gen.pt'))
    netG.eval()
    #netG.cuda()
    #noise = torch.randn(1, nz, lf, lf, lf).cuda()
    noise = torch.randn(1, nz, lf, lf, lf)
    if periodic:
        if periodic[0]: # shravan - for periodicity along X-direction
            noise[:, :, :2] = noise[:, :, -2:]
        if periodic[1]: # shravan - for periodicity along Y-direction
            noise[:, :, :, :2] = noise[:, :, :, -2:]
        if periodic[2]: # shravan - for periodicity along Z-direction
            noise[:, :, :, :, :2] = noise[:, :, :, :, -2:]
    with torch.no_grad():   # shravan - since we are evaluating the model, computation of gradients is not needed, so we turn them off here and just create a raw image from noise using the netG model. It also does: (1) normalisation layers use running statistics (2) de-activates Dropout layers
        raw = netG(noise)   
    print('Postprocessing')
    gb = post_proc(raw,imtype)[0]   # raw has a size of 1X256X192X192X192. post_proc(raw,imtype) has a size of 1X192X192X192 after maxing out the one-hot encoded representation. gb has a size of 192X192X192
    if periodic:
        if periodic[0]:
            gb = gb[:-1]
        if periodic[1]:
            gb = gb[:,:-1]
        if periodic[2]:
            gb = gb[:,:,:-1]  
    tif = np.int_(gb)
    tifffile.imwrite(pth + '_prediction.tif', tif)  # blank images are displayed. could be because of the edgecolor being black
    
    # shravan - plot 3d voxel data
    fig = plt.figure()
    #ax = fig.gca(projection='3d')   # This way is deprecated after matplotlib 3.4
    ax = plt.subplot(projection='3d')
    cmap = plt.get_cmap(colormap_display)
    norm = plt.Normalize(tif.min(), tif.max())
    ax.voxels(np.ones_like(tif), facecolors=cmap(norm(tif)))
    plt.savefig(pth + '_prediction_voxels.png')
    #plt.show()
    
    # shravan - save 5 slices along each direction
    slcs = 5
    fig, axs = plt.subplots(slcs, 3)    # plots the graphs in 5 rows by 3 columns format if slcs=5. ax returns an array of axes
    if imtype == 'colour':
        for j in range(slcs):
            axs[j, 0].imshow(gb[j, :, :, :], vmin = 0, vmax = 255) # shravan - [j,0]--> jth row and 0th column 
            axs[j, 1].imshow(gb[:, j, :, :],  vmin = 0, vmax = 255)    # imshow(data_of_the_image,cmap,vmin=colorbar_min_range,vmax=colorbar_max_range... etc.)
            axs[j, 2].imshow(gb[:, :, j, :],  vmin = 0, vmax = 255)
    elif imtype == 'grayscale':
        for j in range(slcs):
            axs[j, 0].imshow(gb[j, :, :], cmap = 'gray')
            axs[j, 1].imshow(gb[:, j, :], cmap = 'gray')
            axs[j, 2].imshow(gb[:, :, j], cmap = 'gray')
    else:
        for j in range(slcs):
            axs[j, 0].imshow(gb[j, :, :], cmap = colormap_display)  # shravan - by default, the 'viridis' color map is used
            axs[j, 1].imshow(gb[:, j, :], cmap = colormap_display)
            axs[j, 2].imshow(gb[:, :, j], cmap = colormap_display)
    plt.savefig(pth + '_prediction_slices.png')
    plt.close()

    return tif, raw, netG







