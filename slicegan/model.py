from slicegan import preprocessing, util
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import matplotlib

def train(pth, imtype, datatype, real_data, Disc, Gen, nc, l, nz, sf):
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
    :return:
    """
    if len(real_data) == 1:     # shravan - if only one file path is specified
        real_data *= 3  # shravan - if x = [1,2,5] then x *= 3 gives x = [1,2,5,1,2,5,1,2,5]. In other words, the path specified is copied for all three directions 
        isotropic = True
    else:
        isotropic = False

    print('Loading Dataset...')
    dataset_xyz = preprocessing.batch(real_data, datatype, l, sf)   # shravan - dataset_xyz is the list containing the addresses of the data. 1st element for the first image data (one-hot encoded) - tensor of size (32*10,len(phases),l,l), 2nd element for second image data ... etc.

    ## Constants for NNs
    matplotlib.use('Agg')
    ngpu = 1
    num_epochs = 2

    # batch sizes
    batch_size = 8  # shravan - how many samples per batch to load
    D_batch_size = 8
    # optimiser params for G and D
    lrg = 0.0001    # <-- for generator
    lrd = 0.0001    # <-- for discriminator
    beta1 = 0.9     # <-- same betas are used for both generators and discriminators
    beta2 = 0.99
    Lambda = 10     # <-- parameter for gradient penalty
    critic_iters = 5
    cudnn.benchmark = True
    workers = 0
    lz = 4
    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, " will be used.\n")

    # D trained using different data for x, y and z directions
    dataloaderx = torch.utils.data.DataLoader(dataset_xyz[0], batch_size=batch_size,    # dataloaderx is the pointer to object that has a size of (32*10/batch_size). i.e. the data_xyz[0] (having a size of 32*900) is divided into batches of size 8 giving rise to (32*900/8)=3600 batches
                                              shuffle=True, num_workers=workers)        # dataloaderx's first element contains a tensor object of size (batch_size,len(phases),l,l) and so on... up to (32*10/batch_size) elements
    dataloadery = torch.utils.data.DataLoader(dataset_xyz[1], batch_size=batch_size,
                                              shuffle=True, num_workers=workers)
    dataloaderz = torch.utils.data.DataLoader(dataset_xyz[2], batch_size=batch_size,
                                              shuffle=True, num_workers=workers)

    # Create the Genetator network
    netG = Gen().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

    # Define 1 Discriminator and optimizer for each plane in each dimension
    netDs = []
    optDs = []
    for i in range(3):
        netD = Disc()
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netDs.append(netD)
        optDs.append(optim.Adam(netDs[i].parameters(), lr=lrd, betas=(beta1, beta2)))

    disc_real_log = []
    disc_fake_log = []
    gp_log = []
    Wass_log = []

    print("Starting Training Loop...")
    # For each epoch
    start = time.time()
    for epoch in range(num_epochs): # shravan - loop over number of epochs
        # sample data for each direction (shravan - loop over batches of data)
        for i, (datax, datay, dataz) in enumerate(zip(dataloaderx, dataloadery, dataloaderz), 1):   # (1,(datax,datay,dataz)), (2,(datax,datay,dataz)), ..... datax,datay,dataz are tensor objects. 2nd argument in enumerate() is the starting index for the enumerate objects. i.e. instead of starting with 0, it starts with 1 here.
            dataset = [datax, datay, dataz] # shravan - one-hot encoded representation of sampled images (lxl size) from the slices along x,y,z directions. Every time a batch of 8(batch_size) such samples are used. datax has 'batch_size' tensor objects each with 'len(phases)' number of images with each image having lxl size
            ### Initialise
            ### Discriminator
            ## Generate fake image batch with G
            noise = torch.randn(D_batch_size, nz, lz,lz,lz, device=device)  # nz is the number of z channels. random numbers from normal distribution with zero mean and std of 1. nz must be equal to len(phases)? 
            fake_data = netG(noise).detach()    # netG(noise) returns a tensor object. detach() methods sets the required_grad=false and just returns the tensor object to fake_data
            # for each dim (d1, d2 and d3 are used as permutations to make 3D volume into a batch of 2D images)
            for dim, (netD, optimizer, data, d1, d2, d3) in enumerate(
                    zip(netDs, optDs, dataset, [2, 3, 4], [3, 2, 2], [4, 4, 3])):   # d1=(2,3,4), d2=(3,2,4), d3=(4,2,3)
                if isotropic:
                    netD = netDs[0]
                    optimizer = optDs[0]
                netD.zero_grad()    # sets the discriminator gradient to zero.
                ##train on real images
                real_data = data[0].to(device)  # shravan -- <--- samples taken from the input images. .to() Performs Tensor dtype and/or device conversion
                out_real = netD(real_data).view(-1).mean()  # view(-1) flattens the tensor coming from netD(real_data). for ex. a 2x3x4 tensor is flattended to tensor of size 24
                ## train on fake images
                # perform permutation + reshape to turn volume into batch of 2D images to pass to D
                fake_data_perm = fake_data.permute(0, d1, 1, d2, d3).reshape(l * D_batch_size, nc, l, l)
                out_fake = netD(fake_data_perm).mean()
                gradient_penalty = util.calc_gradient_penalty(netD, real_data, fake_data_perm[:batch_size],
                                                                      batch_size, l,
                                                                      device, Lambda, nc)
                disc_cost = out_fake - out_real + gradient_penalty
                disc_cost.backward()
                optimizer.step()
            #logs for plotting
            disc_real_log.append(out_real.item())
            disc_fake_log.append(out_fake.item())
            Wass_log.append(out_real.item() - out_fake.item())
            gp_log.append(gradient_penalty.item())
            ### Generator Training
            if i % int(critic_iters) == 0:
                netG.zero_grad()
                errG = 0
                noise = torch.randn(batch_size, nz, lz,lz,lz, device=device)
                fake = netG(noise)

                for dim, (netD, d1, d2, d3) in enumerate(
                        zip(netDs, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                    if isotropic:
                        #only need one D
                        netD = netDs[0]
                    # permute and reshape to feed to disc
                    fake_data_perm = fake.permute(0, d1, 1, d2, d3).reshape(l * batch_size, nc, l, l)
                    output = netD(fake_data_perm)
                    errG -= output.mean()
                    # Calculate gradients for G
                errG.backward()
                optG.step()

            # Output training stats & show imgs
            if i % 25 == 0:
                netG.eval()
                with torch.no_grad():
                    torch.save(netG.state_dict(), pth + '_Gen.pt')
                    torch.save(netD.state_dict(), pth + '_Disc.pt')
                    noise = torch.randn(1, nz,lz,lz,lz, device=device)
                    img = netG(noise)
                    ###Print progress
                    ## calc ETA
                    steps = len(dataloaderx)
                    util.calc_eta(steps, time.time(), start, i, epoch, num_epochs)
                    ###save example slices
                    util.test_plotter(img, 5, imtype, pth)  # <--- plots the final slices
                    # plotting graphs
                    util.graph_plot([disc_real_log, disc_fake_log], ['real', 'perp'], pth, 'LossGraph')
                    util.graph_plot([Wass_log], ['Wass Distance'], pth, 'WassGraph')
                    util.graph_plot([gp_log], ['Gradient Penalty'], pth, 'GpGraph')
                netG.train()
