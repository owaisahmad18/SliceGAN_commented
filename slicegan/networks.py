import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
def slicegan_nets(pth, Training, imtype, dk,ds,df,dp,gk,gs,gf,gp):
    """
    Define a generator and Discriminator
    :param Training: If training, we save params, if not, we load params from previous.
    This keeps the parameters consistent for older models
    :return:
    """
    #save params
    params = [dk, ds, df, dp, gk, gs, gf, gp]
    # if fresh training, save params
    if Training:
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    # if loading model, load the associated params file
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp  = pickle.load(filehandle)


    # Make nets
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay], gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x):
            for conv,bn in zip(self.convs[:-1],self.bns[:-1]):
                x = F.relu_(bn(conv(x)))
            #use tanh if colour or grayscale, otherwise softmax for one hot encoded
            if imtype in ['grayscale', 'colour']:
                out = 0.5*(torch.tanh(self.convs[-1](x))+1)
            else:
                out = torch.softmax(self.convs[-1](x),1)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x

    return Discriminator, Generator
def slicegan_rc_nets(pth, Training, imtype, dk,ds,df,dp,gk,gs,gf,gp):
    """
    Define a generator and Discriminator
    :param Training: If training, we save params, if not, we load params from previous.
    This keeps the parameters consistent for older models
    :return:
    """
    #save params
    params = [dk, ds, df, dp, gk, gs, gf, gp]
    # if fresh training, save params
    if Training:
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    # if loading model, load the associated params file
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp  = pickle.load(filehandle)


    # Make nets
    class Generator(nn.Module): # shravan - Generator (child) class inheriting nn.Module class (parent class)
        def __init__(self):
            super(Generator, self).__init__()   # shravan - super() function is used to give access to the methods and properties of parent class. It retuns an object that represents the parent class.
            self.convs = nn.ModuleList()    # shravan - convolutions (see output networks given after the code)
            self.bns = nn.ModuleList()      # shravan - batch normalizations    
            self.rcconv = nn.Conv3d(gf[-2],gf[-1],3,1,0)    # shravan - Conv3d(num_of_channels_in_input,num_of_channels_in_output,kernel_size,stride,padding,...) Applies a 3D convolution over an input signal composed of several input planes.
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):   # shravan - number of layers is determined from the size of gk or gs or gp
                self.convs.append(nn.ConvTranspose3d(gf[lay], gf[lay+1], k, s, p, bias=False))  # shravan - ConvTranspose3d(num_of_channels_in_input,num_of_channels_in_output,kernel_size,stride,padding,) Applies a 3D transposed convolution operator over an input image composed of several input planes.
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))  # shravan - normalization of the inputs batch-wise
                # self.bns.append(nn.InstanceNorm3d(gf[lay+1]))

        def forward(self, x):
            for lay, (conv, bn) in enumerate(zip(self.convs[:-1],self.bns[:-1])):
                x = F.relu_(bn(conv(x)))
            size = (int(x.shape[2]-1,)*2,int(x.shape[3]-1,)*2,int(x.shape[3]-1,)*2)
            up = nn.Upsample(size=size, mode='trilinear', align_corners=False)
            out = torch.softmax(self.rcconv(up(x)), 1)
            # print(out.shape)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x

    return Discriminator, Generator
    
# OUTPUT
#netD:  Discriminator(
#  (convs): ModuleList(
#    (0): Conv2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#    (1): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#    (2): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#    (3): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#    (4): Conv2d(512, 1, kernel_size=(4, 4), stride=(2, 2), bias=False)
#  )
#)
#netG:  Generator(
#  (convs): ModuleList(
#    (0): ConvTranspose3d(32, 1024, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(2, 2, 2), bias=False)
#    (1): ConvTranspose3d(1024, 512, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(2, 2, 2), bias=False)
#    (2): ConvTranspose3d(512, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(2, 2, 2), bias=False)
#    (3): ConvTranspose3d(128, 32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(2, 2, 2), bias=False)
#    (4): ConvTranspose3d(32, 256, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
#  )
#  (bns): ModuleList(
#    (0): BatchNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    (2): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    (3): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    (4): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#  )
#  (rcconv): Conv3d(32, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
#)    
    