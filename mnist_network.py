'''
SCAN network on mnist 
Choijilsuren

'''


import torch.nn as nn
from networks import PrimeCapsuleLayer, DigitCapsuleLayer, ConvLayer, NonLocalLayer
from torch.optim import Adam


class MnistSCAN(nn.Module):

    def __init__(self, convlayer, primary_cap, digitcap):

        super(MnistSCAN, self).__init__()

        self.convlayer = ConvLayer()

        self.non_local = NonLocalLayer()
        self.prime_layer = PrimeCapsuleLayer()

        self.digit_layer = DigitCapsuleLayer()
        
        
    def forward(self, x):

        conv_out = self.convlayer(x)
        nonlocal_out = self.non_local(conv_out)

        prime_out = self.prime_layer(nonlocal_out)
        digit_out = self.digit_layer(prime_out)

        reconst = self.recont_layer(digit_out)

        return reconst, digit_out
        
    
    def loss(self, ):

        pass

    
    #margin loss 
    def margin_loss(self, data, x,):

        pass 


    def reconst_loss(self, ):

        pass

    

if __name__ == '__main__':

    pass


