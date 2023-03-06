import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
def batch(data,type,l, sf,nSamplesFromRealImages):
    """
    Generate a batch of images randomly sampled from a training microstructure
    :param data: data path -- array containing the paths to images
    :param type: data type
    :param l: image size
    :param sf: scale factor
    :return:
    """
    Testing = False
    if type in ['png', 'jpg', 'tif2D']:
        datasetxyz = []
        for img in data:        # shravan -- data is the array containing the paths to images
            img = plt.imread(img) if type != 'tif2D' else tifffile.imread(img)
            if len(img.shape)>2:    # img = [[[1,2,3],[34,56,78],..........256Values], [[],[],.......256Values]..................256values]
                img = img[:,:,0]    # shravan - gets the Red channel value for all the pixels in the image. :,:,0 --> 256(rows),256(columns),0th entry (red pixel value) for the row and column
            img = img[::sf,::sf]    # shravan - image size becomes l/sf. if l=64 and sf=2, after scaling original image of size 64x64 pixels becomes 32x32 pixels
            x_max, y_max= img.shape[:]
            phases = np.unique(img)  # shravan - img is the red channel value for all the pixels i.e. conversion of red color into gray scale. each unique value in this space represents a unique phase
            data = np.empty([nSamplesFromRealImages, len(phases), l, l])  # creates a matrix of dimensions (32*10,len(phases),l,l) -- len(phases), lt should be resonably small, otherwise memory issues happen. for the NMC data, there are three phases white, black and gray. Number of batches would be : (first argument here/batch size)
            # shravan - (1) We are sampling 32*10 images (each with size lxl) from the input image
            #           (2) The number of unique phases in the image are determined
            #           (3) One-hot encoding takes one image (in this case a sampled image of size lxl) and assigns 'one' to  pixels with the specific phase and 'zero' to all the other pixels. This is repeated for all the phases
            #           (4) So a single (l x l) image becomes 'nPhases' number of images in its one-hot encoding
            print('Estimated memory for the samples from real image(s): ',((nSamplesFromRealImages)*len(phases)*l*l)*8/1E9, 'GB')        # assuming 8bytes for a floating number
            for i in range(nSamplesFromRealImages):    # shravan - sample 32*10 images
                x = np.random.randint(1, x_max - l-1)   # shravan - generate a random integer between 1 and x_max-l-1
                y = np.random.randint(1, y_max - l-1)   # shravan - x,y here represent the random starting positions of the pixels of image with size lxl. Since the image to be sampled can't go beyond the input image boundary, (x,y) can only be between (1,x_max-l-1) and (1,y_max-l-1). if (x,y)=(x_max,y_max) then we can still sample an image of size lxl without going out of the input image
                # create one channel per phase for one hot encoding
                for cnt, phs in enumerate(phases):  # enumerate --> [(0,phases[0]),(1,phases[1]) .....] <--- one-hot encoding of the sampled image
                    img1 = np.zeros([l, l]) # shravan - matrix of size[l,l]. elements can be accessed with img1[i,j]
                    img1[img[x:x + l, y:y + l] == phs] = 1  # shravan - samples image --> img[x:x+l,y:y+l]. if any of the pixels in this image has phase value of phs, then it is assigned 1. else they are assigned zero. 
                    data[i, cnt, :, :] = img1   # shravan -- <-- 'i'th sampled image, 'nPhases' images with each one represeting the one-hot encoded representation for each of the nPhases, l, l

            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :]+2*data[j, 1, :, :]) # shravan - for the first 8 sampled images, take first two phases and combine them with linear combination
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)  # shravan - creates a torch tensor (multi-dimensional matrix) from numpy arrays or python lists -- here 32*10,len(phases),l,l size
            dataset = torch.utils.data.TensorDataset(data)  # shravan - tensor 'data' is converted into some sort of pointer. i.e. dataset has the address of the object containing 'data'
            datasetxyz.append(dataset)  # shravan - datasetxyz is the list containing the addresses of the data. 1st element for the first image data (one-hot encoded), 2nd element for second image data ... etc.

    elif type=='tif3D':
        datasetxyz=[]
        img = np.array(tifffile.imread(data[0]))
        img = img[::sf,::sf,::sf]
        ## Create a data store and add random samples from the full image
        x_max, y_max, z_max = img.shape[:]
        print('training image shape: ', img.shape)
        vals = np.unique(img)   # shravan - sorted unique elements of array img
        for dim in range(3):
            data = np.empty([nSamplesFromRealImages, len(vals), l, l])
            print('dataset ', dim)
            for i in range(32*900):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                z = np.random.randint(0, z_max - l)
                # create one channel per phase for one hot encoding
                lay = np.random.randint(img.shape[dim]-1)
                for cnt,phs in enumerate(list(vals)):   # shravan - cnt is the loop counter and phs is the actual entry in vals
                    img1 = np.zeros([l,l])
                    if dim==0:
                        img1[img[lay, y:y + l, z:z + l] == phs] = 1
                    elif dim==1:
                        img1[img[x:x + l,lay, z:z + l] == phs] = 1
                    else:
                        img1[img[x:x + l, y:y + l,lay] == phs] = 1
                    data[i, cnt, :, :] = img1[:,:]
                    # data[i, (cnt+1)%3, :, :] = img1[:,:]

            if Testing:
                for j in range(2):
                    plt.imshow(data[j, 0, :, :] + 2 * data[j, 1, :, :])
                    plt.pause(1)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='colour':
        ## Create a data store and add random samples from the full image
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            img = img[::sf,::sf,:]
            ep_sz = nSamplesFromRealImages
            data = np.empty([ep_sz, 3, l, l])
            x_max, y_max = img.shape[:2]
            for i in range(ep_sz):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                # create one channel per phase for one hot encoding
                data[i, 0, :, :] = img[x:x + l, y:y + l,0]
                data[i, 1, :, :] = img[x:x + l, y:y + l,1]
                data[i, 2, :, :] = img[x:x + l, y:y + l,2]
            print('converting')
            if Testing:
                datatest = np.swapaxes(data,1,3)
                datatest = np.swapaxes(datatest,1,2)
                for j in range(5):
                    plt.imshow(datatest[j, :, :, :])
                    plt.pause(0.5)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='grayscale':
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = img/img.max()
            img = img[::sf, ::sf]
            x_max, y_max = img.shape[:]
            data = np.empty([nSamplesFromRealImages, 1, l, l])
            for i in range(nSamplesFromRealImages):
                x = np.random.randint(1, x_max - l - 1)
                y = np.random.randint(1, y_max - l - 1)
                subim = img[x:x + l, y:y + l]
                data[i, 0, :, :] = subim
            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)
            
    return datasetxyz


