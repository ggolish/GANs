#!/usr/bin/env python3
import pytorch
from architecture import DC

from pytorch.nn import Module


class Critic(Module):

    def __init__(self, arch):
        super().__init__()
        self.arch = arch

    def forward(self, x):
        return self.arch(x)

class Generator(Module):

    def __init__(self, arch):
        super().__init__()
        self.arch = arch

    def forward(self, x):
        return self.arch.forward(x)

if __name__ == '__main__':
    C = Critic(DC(True))
    D = Critic(DC(False))
