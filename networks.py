
'''
SACN --> Self-Attention Capsule Network

'''


import torch.nn as nn
import torch

from torch.autograd import Variable
import torch.nn.functional as F


USE_CUDA = True if torch.cuda.is_available() else False

class PrimeCapsuleLayer(nn.Module):

    def __init__(self, caps_dim = 8, in_channel = 256, out_channel = 32,
                 num_routes = 32*6*6, kernel_size = 9):

        super(PrimeCapsuleLayer, self).__init__()
        
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels = in_channel , out_channels = out_channel, kernel_size = kernel_size,
                      stride = 2, padding = 0)
            for _ in range(caps_dim)])
    
        self.num_routes = num_routes


    def forward(self, x):

        batch_size = x.size(0)
        
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)

        u = u.view(batch_size, self.num_routes, -1)

        return self.squash(u)

    def squash(self, input_tensor):

        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor/((1. + squared_norm) *torch.sqrt(squared_norm))
        return output_tensor
        
        
class DigitCapsuleLayer(nn.Module):

    def __init__(self, num_capsule = 10, num_routes = 32*6*6, incap_dim = 8, outcap_dim = 16):
    
        super(DigitCapsuleLayer, self).__init__()

        self.num_capsule = num_capsule
        self.num_routes = num_routes

        self.incap_dim = incap_dim
        self.outcap_dim = outcap_dim

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsule, outcap_dim, incap_dim))
        
    def squash(self, input_tensor):

        squared_norm = (input_tensor ** 2).sum(-1, keepdim = True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm)*torch.sqrt(squared_norm))
        return output_tensor

    
    def forward(self, x):

        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsule, dim=2).unsqueeze(4)
        
        W = torch.cat([self.W] * batch_size, dim=0)
        
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsule, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 1 
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)
        

class ConvLayer(nn.Module):

    def __init__(self, input_channel=1, out_channel=256, kernel_size=9,
                 stride = 1):

        super(ConvLayer, self).__init__()
        
        self.conv = nn.Conv2d(input_channel, out_channel, kernel_size, stride)

    def forward(self,x):

        return F.relu(self.conv(x))


    
class NonLocalLayer(nn.Module):

    def __init__(self, in_channel = 256, inter_channel = 128 , out_channel = 256):

        super(NonLocalLayer, self).__init__()

        #middle channel
        self.inter_channel = inter_channel

        #1x1 convolutions three parralel
        self.f = nn.Conv2d(in_channel, inter_channel, kernel_size = 1,
                           stride=1)
        self.g = nn.Conv2d(in_channel, inter_channel, kernel_size =1,
                           stride = 1)
        self.h = nn.Conv2d(in_channel, inter_channel,kernel_size = 1,
                           stride = 1)

        self.alpha = torch.nn.Parameter(torch.zeros(1))
        
        #self.softmax_2d = nn.Softmax(dim=1)
        self.conv_layer = nn.Conv2d(inter_channel, out_channel, kernel_size=1,
                                    stride=1)
        
        
        
    def forward(self,x):

        batch_size = x.size(0)

        #
        f_x = self.f(x).view(batch_size,  self.inter_channel, -1)

        #
        g_x = self.g(x).view(batch_size, self.inter_channel, -1)
        g_x = g_x.permute(0, 2, 1 )

        fg = torch.matmul(f_x, g_x)
        fg = F.softmax(fg)

        h_x = self.h(x).view(batch_size, self.inter_channel, -1)
        fgh = torch.matmul(fg, h_x).view(batch_size,  self.inter_channel,
                                         *x.size()[2:])

        fgh = self.conv_layer(fgh)*self.alpha
        return fgh + x

    
class Reconstruction(nn.Module):

    def __init__(self, input_height, input_width, input_channel):

        super(Reconstruction, self).__init__()

        self.input_channel = input_channel
        self.input_width = input_width
        self.input_height = input_height
        
        self.networks = nn.Sequential(
            nn.Linear(10*16, 512),
            nn.ReLU(inplace = True),
            nn.Linear(512,1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, input_height*input_width*input_channel),
            nn.Sigmoid()
        )
        
            
    def forward(self, x, data):

        classes = torch.sqrt((x**2).sum(2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))

        if USE_CUDA:
            masked = masked.cuda()

            
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.networks(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked
        

    



    


    
