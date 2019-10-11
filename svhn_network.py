'''
All parts still same as MNIST -> but whole network specifics are little bit 
different

'''

import numpy as np
import torch.nn as nn

from torch.optim import Adam
import torch
from torch.autograd import Variable
from dataloader import getSVHN_Loader
import torch.nn.functional as F

from networks import PrimeCapsuleLayer, DigitCapsuleLayer, ConvLayer, NonLocalLayer, Reconstruction


class SVHN(nn.Module):

    def __init__(self):
        
        
        super(MnistSCAN, self).__init__()

        self.conv_layer = ConvLayer(input_channel=)

        self.nonlocal_layer = NonLocalLayer()
        self.prime_layer = PrimeCapsuleLayer()
        self.digit_layer = DigitCapsuleLayer()

        
        
    def forward(self, x):

        conv_out = F.relu(self.conv_layer(x))
        nonlocal_out = self.nonlocal_layer(conv_out)

        prime_out = self.prime_layer(conv_out)
        
        
    def loss(self):

        pass

    def margin_loss(self):

        pass


    def reconst_loss(self, x):

        pass


    
        
