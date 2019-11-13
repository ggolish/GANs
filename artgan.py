#!/usr/bin/env python3
import pytorch
import architecture

from pytorch.nn import Module


class Critic(Module):

    def __init__(self, arch):
        super().__init__()
        self.arch = arch

    def forward(self, x):
        return self.arch.forward(x)


class Generator(Module):

    def __init__(self, arch):
        super().__init__()
        self.arch = arch

    def forward(self, x):
        return self.arch.forward(x)
