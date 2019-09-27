'''
SCAN network on mnist 
Choijilsuren

'''


import torch.nn as nn
from networks import PrimeCapsuleLayer, DigitCapsuleLayer, ConvLayer, NonLocalLayer
from torch.optim import Adam


class MnistSCAN(nn.Module):

    def __init__(self):

        super(MnistSCAN, self).__init__()

        self.convlayer = ConvLayer(input_channel=1, out_channels=256,
                                   kernel_size=9, stride=1)

        self.non_local = NonLocalLayer(in_channel=256, inter_channel=128,
                                       out_channel=256)
        
        self.prime_layer = PrimeCapsuleLayer(caps_dim=8, in_channel=256,
                                             out_channel=32, num_routes=32*6*6,
                                             kernel_size=9)
        
        self.digit_layer = DigitCapsuleLayer(num_capsule=10, num_routes=32*6*6,
                                             incap_dim=8, outcap_dim=16)
                
        
    def forward(self, x):

        conv_out = self.convlayer(x)
        nonlocal_out = self.non_local(conv_out)

        prime_out = self.prime_layer(nonlocal_out)
        digit_out = self.digit_layer(prime_out)

        reconst = self.recont_layer(digit_out)

        return reconst, digit_out
        
    
   def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)     

    
    #margin loss 
    def margin_loss(self, labels, x):

        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss


    def reconst_loss(self, ):

         loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1),
                              data.view(reconstructions.size(0), -1))
         
        return loss * 0.0005

    

if __name__ == '__main__':

    mnist_network = MnistSCAN()

    

