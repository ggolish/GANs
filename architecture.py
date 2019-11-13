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

    def __init__(self, iscritic=True):
        super().__init__(iscritic)
        
    def forward(self, x):
        pass


# Experimental ResNet architecture (have not seen this in a paper yet)
class ResNet(Architecture):

    def __init__(self, iscritic=True):
        super().__init__(iscritic)
        
    def forward(self, x):
        pass


# Simple multilayer perceptron architecture for comparsion purposes
class MLP(Architecture):

    def __init__(self, iscritic=True):
        super().__init__(iscritic)
        
    def forward(self, x):
        pass


