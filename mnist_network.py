'''
SCAN network on mnist 
Choijilsuren

'''


import torch.nn as nn

from torch.optim import Adam

import torch
from torch.autograd import Variable

from dataloader import getMNIST_Loader
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import numpy as np 

from networks import PrimeCapsuleLayer, DigitCapsuleLayer, ConvLayer, NonLocalLayer, Reconstruction

USE_CUDA = torch.cuda.is_available()

writer = SummaryWriter()

class MnistSCAN(nn.Module):

    def __init__(self):

        super(MnistSCAN, self).__init__()

        self.convlayer = ConvLayer(input_channel=1, out_channel=256,
                                   kernel_size=9, stride=1)
        
        self.non_local = NonLocalLayer(in_channel=256, inter_channel=128,
                                       out_channel=256)
        
        self.prime_layer = PrimeCapsuleLayer(caps_dim=8, in_channel=256,
                                             out_channel=32, num_routes=32*6*6,
                                             kernel_size=9)
        
        self.digit_layer = DigitCapsuleLayer(num_capsule=10, num_routes=32*6*6,
                                             incap_dim=8, outcap_dim=16)
                
        self.recon_layer = Reconstruction(input_height=28, input_width=28,
                                          input_channel=1)

        self.mse_loss = nn.MSELoss()
        
    def forward(self, x):

        conv_out = F.relu(self.convlayer(x))
        nonlocal_out = self.non_local(conv_out)

        prime_out = self.prime_layer(nonlocal_out)
        digit_out = self.digit_layer(prime_out)

        reconst, masked  = self.recon_layer(digit_out, x)

        return  digit_out, reconst, masked
        
    
    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconst_loss(data, reconstructions)     

    
    #margin loss 
    def margin_loss(self, x, labels):

        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        
        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    
    def reconst_loss(self, data, reconstructions):

         loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1),
                              data.view(reconstructions.size(0), -1))
         
         return loss * 0.392


def train(train_loader, optimizer, model, epoch):

    train_loss = 0.0
    
    for i,  (data, target) in enumerate(train_loader):
        
        #one hot         
        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)
            
        batch_size = data.size(0)
        length_data = len(train_loader.dataset)
        
        if USE_CUDA:
                
            data = data.cuda()
            target = target.cuda()
            model = model.cuda()

        output, reconstructions, masked = model(data)
        loss = model.loss(data, output, target, reconstructions)
            
        loss.backward()
        optimizer.step()
        
        correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        
        train_loss = loss.data
        n_iter = int(length_data)*epoch + i
        
        writer.add_scalar('Loss/Train', train_loss, n_iter)
        writer.add_scalar('Accuracy/Train', correct/float(batch_size), n_iter)
        
        if i % 100 == 0:
                
            print ('Epoch Number {}: train_accuracy: {} loss: {}'.format(i, correct/float(batch_size),
                                                                         train_loss))
            
            
def test(test_loader, model, epoch):

    model = model.eval()

    test_loss = 0.0
    correct = 0
        
    for i, (test_data, test_target) in enumerate(test_loader):

        test_target = torch.sparse.torch.eye(10).index_select(dim=0, index=test_target)
        data, target = Variable(test_data), Variable(test_target)

        batch_size = data.size(0)
        length_data = len(test_loader.dataset)
        
        if USE_CUDA:
            data = data.cuda()
            target = target.cuda()
            model = model.cuda()

        output, reconstructions, masked = model(data)

        with torch.no_grad():
            loss = model.loss(data, output, target, reconstructions)

            test_loss = loss.data

            cor = sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                           np.argmax(target.data.cpu().numpy(), 1))

            
            correct += sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                           np.argmax(target.data.cpu().numpy(), 1))

        n_iter = epoch*int(length_data) + i

        writer.add_scalar('Loss/Test', test_loss, n_iter)
        writer.add_scalar('Accuracy/Test', cor/float(batch_size), n_iter)

    print ('test accuracy: {} test_loss: {}'.format(float(correct)/float(length_data),
                                                    float(test_loss)/float(length_data)))

           
if __name__ == '__main__':

    mnist_network = MnistSCAN()

    train_loader, test_loader = getMNIST_Loader(batch_size = 100)

    optimizer = torch.optim.Adam(mnist_network.parameters(), lr=0.0001)

    for e in range(10):
        train(train_loader, optimizer, mnist_network, e)
        test(test_loader, mnist_network, e)
    
