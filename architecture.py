#!/usr/bin/env python3
'''
    Differenet architectures to be used as the generator or critic in a WGAN.
    The architectures are specific to the data being used in our experiments.
'''

# Abstract class to ensure all architectures take the iscritic argument and 
# implement the forward method
class Architecture():
     
    def __init__(self, iscritic=True):
        self.iscritic = iscritic

    def forward(self, x):
        return x


# The architecture described in the DCGAN paper
class DeepConvolution(Architecture):

    def __init__(self, nz, ngf, nc iscritic=True):
        super().__init__(iscritic)
        
    def forward(self, x):
        if self.critic:
            return self.critic(x)
        else:
            return self.generator(x)

    def critic(self, x):
        return nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

def DC(x, isCritic):
    if isCritic:
        pass
    else:
        return nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )



# Experimental ResNet architecture (have not seen this in a paper yet)
class ResNet(Architecture):

    def __init__(self, iscritic=True):
        super().__init__(iscritic)
        
    def forward(self, x):
        pass

# Simple multilayer perceptron architecture for comparsion purposes
class MLPerceptron(Architecture):

    def __init__(self, iscritic=True):
        super().__init__(iscritic)
        
    def forward(self, x):
        pass


