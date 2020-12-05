import torch
import torch.nn as nn
import torch.nn.functional as F
from gan.spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=4, stride=2, padding = 1))
        self.conv2 = SpectralNorm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding = 1))
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding = 1))
        self.conv4 = SpectralNorm(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding = 1))
        self.conv5 = SpectralNorm(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1))
        self.activation = nn.LeakyReLU(negative_slope=.2)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        # print('D forward')
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.activation((self.conv5(x)))
        assert list(x.shape[1:]) == [1, 1, 1]
        
        ##########       END      ##########

        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv1 = nn.ConvTranspose2d(in_channels=noise_dim, out_channels=1024,            kernel_size=4, stride=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=1024,      out_channels=512,             kernel_size=4, stride=2, padding = 1)
        self.conv3 = nn.ConvTranspose2d(in_channels=512,       out_channels=256,             kernel_size=4, stride=2, padding = 1)
        self.conv4 = nn.ConvTranspose2d(in_channels=256,       out_channels=128,             kernel_size=4, stride=2, padding = 1)
        self.conv5 = nn.ConvTranspose2d(in_channels=128,       out_channels=output_channels, kernel_size=4, stride=2, padding = 1)

        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)

        self.activation = nn.LeakyReLU(negative_slope=.2)
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        # print('G forward')
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = torch.tanh((self.conv5(x)))
        assert list(x.shape) == [x.shape[0], 3, 64, 64]


        ##########       END      ##########
        return x
    

