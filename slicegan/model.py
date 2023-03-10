from slicegan import preprocessing, util
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import matplotlib

def train(pth, imtype, datatype, real_data, Disc, Gen, nc, l, nz, sf,nBatchesBeforeUpdatingGenerator,nSamplesFromRealImages,ngpu,num_epochs,batch_size,D_batch_size,lrg,lrd,beta1,beta2,Lambda,lz,workers):
    """
    train the generator
    :param pth: path to save all files, imgs and data
    :param imtype: image type e.g nphase, colour or gray
    :param datatype: training data format e.g. tif, jpg ect
    :param real_data: path to training data
    :param Disc:
    :param Gen:
    :param nc: channels
    :param l: image size
    :param nz: latent vector size
    :param sf: scale factor for training data
    :param lz: latent space size (size of white-noise image used as input to netG)
    :return:
    """
    if len(real_data) == 1:     # shravan - if only one file path is specified
        real_data *= 3  # shravan - if x = [1,2,5] then x *= 3 gives x = [1,2,5,1,2,5,1,2,5]=[x_orig x_orig x_orig]. In other words, the path specified is copied for all three directions 
        isotropic = True
    else:
        isotropic = False

    print('Loading Dataset...')
    dataset_xyz = preprocessing.batch(real_data, datatype, l, sf,nSamplesFromRealImages)   # shravan - dataset_xyz is the list containing the addresses of the data. This is a tensor of size 3X(32*10,len(phases),l,l). 1st element for the first reference image data (one-hot encoded) of size len(phases)XlXl, 2nd element for second reference image data ... etc. The same applies for other two dimesnions too.
    
    matplotlib.use('Agg')
    critic_iters = nBatchesBeforeUpdatingGenerator  #5
    cudnn.benchmark = True      # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware. Benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime. But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.
    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")   # A torch.device is an object representing the device on which a torch.Tensor is or will be allocated. The torch.device contains a device type ('cpu' or 'cuda') and optional device ordinal for the device type. If the device ordinal is not present, this object will always represent the current device for the device type, even after torch.cuda.set_device() is called; e.g., a torch.Tensor constructed with device 'cuda' is equivalent to 'cuda:X' where X is the result of torch.cuda.current_device().
    print(device, " will be used.\n")

    # D trained using different data for x, y and z directions
    dataloaderx = torch.utils.data.DataLoader(dataset_xyz[0], batch_size=batch_size,    # dataloaderx is the pointer to object that has a size of (32*10/batch_size). i.e. the data_xyz[0] (having a size of 32*10) is divided into batches of size 8 giving rise to (32*10/8)=40 batches
                                              shuffle=True, num_workers=workers)        # dataloaderx's first element contains a tensor object of size (batch_size,len(phases),l,l) and so on... up to (32*10/batch_size) elements
    dataloadery = torch.utils.data.DataLoader(dataset_xyz[1], batch_size=batch_size,
                                              shuffle=True, num_workers=workers)
    dataloaderz = torch.utils.data.DataLoader(dataset_xyz[2], batch_size=batch_size,
                                              shuffle=True, num_workers=workers)

    # Create the Genetator network
    netG = Gen().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))  # shravan - lrg is the learning rate for the generator network

    # Define 1 Discriminator and optimizer for each plane in each dimension
    netDs = []
    optDs = []
    for i in range(3):
        netD = Disc()
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netDs.append(netD)
        optDs.append(optim.Adam(netDs[i].parameters(), lr=lrd, betas=(beta1, beta2)))       # shravan - lrd is the learning rate for the discriminator network

    disc_real_log = []  # shravan - discriminator loss for classifying the real images
    disc_fake_log = []  # shravan - discriminator loss for classifying the fake images
    gp_log = []         # shravan - gradient penalty loss
    Wass_log = []       # shravan - Wasserstein loss

    print("Starting Training Loop...")
    # For each epoch
    start = time.time()
    for epoch in range(num_epochs): # shravan --------------------------------------------- loop over number of epochs
        # sample data for each direction (shravan ---------------------------------------------- loop over batches of data)
        for i, (datax, datay, dataz) in enumerate(zip(dataloaderx, dataloadery, dataloaderz), 1):   # (1,(datax,datay,dataz)), (2,(datax,datay,dataz)), ..... datax,datay,dataz are tensor objects. 2nd argument in enumerate() is the starting index for the enumerate objects. i.e. instead of starting with 0, it starts with 1 here. 1,2,3... etc. are the batch numbers.
            dataset = [datax, datay, dataz] # shravan - size of dataset is nDimensionsX1Xbatch_sizeXnPhasesXimg_size(l)Ximg_size(l). These are one-hot encoded representation of sampled images (lxl size) from the slices along x,y,z directions. Every time a batch of 8(batch_size) such samples are used. datax has 'batch_size' tensor objects each with 'len(phases)' number of images with each image having lxl size
            ### Initialise
            ### Discriminator
            ## Generate fake image batch with G
            noise = torch.randn(D_batch_size, nz, lz,lz,lz, device=device)  # nz is the number of z channels. random numbers from normal distribution with zero mean and std of 1. nz is the number of phases used for the noise image for its one-hot encoding representation. nz does not have to be equal to len(phases). 
            #print('netGoutput-shape: ',netG(noise).size())
            fake_data = netG(noise).detach()    # netG(noise) returns a tensor object of size batch_sizeXnPhasesXimg_size(l)Ximg_size(l)Ximg_size(l). detach() methods sets the required_grad=false and just returns the tensor object to fake_data
                                                # i.e. netG() generates batch_size number of 3D-volumes in their one-hot encoded representation.
            # ---------------------------------------  loop over each dimension i.e. X,Y and Z
            for dim, (netD, optimizer, data, d1, d2, d3) in enumerate(              # if A=[11,56,78], then enumerate(A) gives 0-->11, 1-->56, 2-->78 etc.
                    zip(netDs, optDs, dataset, [2, 3, 4], [3, 2, 2], [4, 4, 3])):   # The list 'data' has real-image-data and has a size of 1Xbatch_sizeXnPhasesXimg_size(l)Ximg_size(l)
                if isotropic:                                           # dim=0 --> netD=netDs[0],optimizer=optDs[0],data=dataset[0],d1=2,d2=3,d3=4
                    netD = netDs[0]                                     # dim=1 --> netD=netDs[1],optimizer=optDs[1],data=dataset[1],d1=3,d2=2,d3=4
                    optimizer = optDs[0]                                # dim=2 --> netD=netDs[2],optimizer=optDs[2],data=dataset[2],d1=4,d2=2,d3=3
                netD.zero_grad()    # sets the discriminator gradient to zero.
                ##train on real images
                # data[0] has a size of batch_sizeXnPhasesXimg_size(l)Ximg_size(l)
                # i.e. sample batch_size number of images (their one-hot-encoded representation) of size img_size(l)Ximg_size(l) from the input image for comparison with fake images (of same size) later
                real_data = data[0].to(device)  # shravan -- <--- samples taken from the input images. .to() Performs Tensor dtype and/or device conversion. data[0] has a size of batch_sizeXnPhasesXimg_size(l)Ximg_size(l)
                # netD(real_data) has a size of batch_sizeXoneXoneXone
                out_real = netD(real_data).view(-1).mean()  # netD(real_data) has a size of [batch_size,1,1,1]. view(-1) flattens the tensor coming from netD(real_data). for ex. a 2x3x4 tensor is flattended to tensor of size 24
                ## train on fake images
                # perform permutation + reshape to turn volume into batch of 2D images to pass to D                
                # For:: dim=0, permute the indices (0,1,2,3,4) of fake_data to (0,2,1,3,4). i.e. the 
                # size batch_sizeXnPhasesXimg_size(l)Ximg_size(l)Ximg_size(l) becomes batch_sizeXimg_size(1)Xn_phasesXimg_size(l)Ximg_size(l)
                # for ex: batch_size=4, nPhase=nc=256, img_size=l=64, i.e. size 4X256X64X64X64 becomes 4X64X256X64X64
                # which is reshaped to (64*4,256,64,64) --> one hot-ended represented images (256 phases) of 64 slices (of 4 batches -->64*4) along X-direction each having a size of 64X64 (Y and Z dimension pixels)
                fake_data_perm = fake_data.permute(0, d1, 1, d2, d3).reshape(l * D_batch_size, nc, l, l) # nc = no of channels in the input image, 
                # therefore, netD for this dimension is supplied with 256 images i.e. 64 images along this direction for 4 batches
                out_fake = netD(fake_data_perm).mean()  # netD(fake_data_perm) has a size of img_size(l)*batch_sizeX1X1X1. These are the 64 pixel values for the 4 batches of data.
                                                        # If the mean of the pixel values for the four batches is low, then it is a fake image.

                gradient_penalty = util.calc_gradient_penalty(netD, real_data, fake_data_perm[:batch_size],
                                                                      batch_size, l,
                                                                      device, Lambda, nc)
                disc_cost = out_fake - out_real + gradient_penalty  # shravan -- discriminator cost : out_fake = mean prob of classifying the fake data as real, out_real=mean prob of calssifying the real data as real
                disc_cost.backward()                                # minimization of disc_cost is max imization of (out_real - out_fake). this happens when out_real is as high as possible (i.e. the prob of classifying the real data as real should be high) and out_fake is as low as possible (i.e. the prob of classifying the fake data as real should be small)
                optimizer.step()                                        # if we label the data as: real_data --> 1 and fake_data --> 0, this can be interpreted as the 'probability of data being real' which is the output from discriminator
                                                                              # However, in the Wasserstein loss, the output can have arbitrary number (not necessarily between 0 and 1), in this case, out_real must be as high as possible nd out_fake must be as small as possible (the interpretation is similar to the probabilistic interpretation given above). The dicriminator in this case is some times called 'critic' because it gives high value (happens when out_real >> out_fake) if the classification is correct and smaller value if the classification is bad (happens when out_fake~out_real). see Wasserstein loss definition.
            # --------------------------- end of training of 3 discriminators ------------------------------------
            #logs for plotting
            disc_real_log.append(out_real.item())       # item() returns the entry from tensor of rank 1 as a standard python number. out_real must be a tensor containing just 1 element
            disc_fake_log.append(out_fake.item())
            Wass_log.append(out_real.item() - out_fake.item())
            gp_log.append(gradient_penalty.item())
            ### Generator Training
            isTrainedGeneratorThisBatch = False
            if i % int(critic_iters) == 0:
                isTrainedGeneratorThisBatch = True
                netG.zero_grad()
                errG = 0
                noise = torch.randn(batch_size, nz, lz,lz,lz, device=device)
                fake = netG(noise)
                #print('3D image: ',fake)

                for dim, (netD, d1, d2, d3) in enumerate(
                        zip(netDs, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                    if isotropic:
                        #only need one D
                        netD = netDs[0]
                    # permute and reshape to feed to disc
                    fake_data_perm = fake.permute(0, d1, 1, d2, d3).reshape(l * batch_size, nc, l, l)
                    output = netD(fake_data_perm)   # output is the probability of data supplied is real
                    errG -= output.mean()       # shravan - generator cost = -output from discriminator for the generated fake data; min of generator cost implies maximization of discriminator output mean. i.e. the mean 'probability of data being real' should be high. i.e. the generated sample should be close to real data. i.e. we are tricking the dicriminator.                    
                    # Calculate gradients for G         
                errG.backward() # errG has the error from all the 3 dimensions, i.e. errG = errG - output.mean() is performed over the output of three discriminators
                optG.step()

            # Output training stats & show imgs
            if i % 1 == 0:  # shravan - print the state of networks and images after every batch of training.
                netG.eval()
                with torch.no_grad():
                    torch.save(netG.state_dict(), pth + '_Gen.pt')
                    torch.save(netD.state_dict(), pth + '_Disc.pt')
                    noise = torch.randn(1, nz,lz,lz,lz, device=device)
                    img = netG(noise)   # based on the optimized Generator so far, generate a new image. img has a size of 1XnPhasesXimg_size(l)Ximg_size(l)Ximg_size(l)
                    ###Print progress
                    ## calc ETA
                    steps = len(dataloaderx)    # total number of batches
                    util.calc_eta(steps, time.time(), start, i, epoch, num_epochs,isTrainedGeneratorThisBatch)  # i is the batch number, steps is the total number of batches, In each epoch, all batches are passed to neural network for training.
                    ###save example slices
                    # img is a 3D volume consisting of batch_size number of images in their on-hot-encoded representation.
                    util.test_plotter(img, 5, imtype, pth)  # <--- plots the final slices. imtype can be one of colour, grayscale, three-phase or two-phase or nphases.
                    # plotting graphs
                    util.graph_plot([disc_real_log, disc_fake_log], ['real as real \n(~Prob. of classifying real image as real - should be high)', 'fake as real \n(~Prob. of classifying fake image as real - should be low)'], pth, 'LossGraph','Cumulative batch number','discriminator output')
                    util.graph_plot([Wass_log], ['Wass Distance'], pth, 'WassGraph','Cumulative batch number','\n Wasserstein loss \n(DiscOutputForRealAsReal - DiscOutputForFakeAsReal)' )
                    util.graph_plot([gp_log], ['Gradient Penalty'], pth, 'GpGraph','Cumulative batch number','Gradient penalty')
                netG.train()
