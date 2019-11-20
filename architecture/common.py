
from torch.nn import Module

class Identity(Module):
    ''' Identity layer to act as placeholder in dynamic architectures  '''

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
