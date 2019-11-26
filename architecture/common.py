#!/usr/bin/python3

import sys

if __name__ == 'architecture.common':
    from . import dc
else:
    import dc


def load_generator(settings: dict):
    ''' Returns the appropriate generator based in arch_string '''
    arch_module = arch_string_to_module(settings['generator_arch'])
    return arch_module.GeneratorArchitecture(settings)


def load_critic(settings: dict):
    ''' Returns the appropriate critic based in arch_string '''
    arch_module = arch_string_to_module(settings['critic_arch'])
    return arch_module.CriticArchitecture(settings)


def arch_string_to_module(arch_string: str):
    ''' Returns the appropriate architecture module based on arch_string '''
    if arch_string == 'dc':
        return dc
    else:
        sys.stderr.write(f'Error: Invalid architecture {arch_string}!\n')
        sys.exit(1)
