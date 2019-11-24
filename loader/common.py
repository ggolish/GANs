
import sys

if __name__ == "loader.common":
    from . import ross
    from . import cubism
    from . import impressionism
    from . import cifar
    from . import mnist
else:
    import ross
    import cubism
    import impressionism
    import cifar
    import mnist


def load_dataset(ds: str, optimize=True, imsize=64, batch_size=128, verbose=True):
    if ds == "ross":
        return ross.load(optimize=optimize, imsize=imsize, batch_size=batch_size, verbose=verbose)
    elif ds == "cubism":
        return cubism.load(optimize=optimize, imsize=imsize, batch_size=batch_size, verbose=verbose)
    elif ds == "impressionism":
        return impressionism.load(optimize=optimize, imsize=imsize, batch_size=batch_size, verbose=verbose)
    elif ds == "cifar":
        return cifar.load(optimize=optimize, imsize=imsize, batch_size=batch_size, verbose=verbose)
    elif ds == "mnist":
        return mnist.load(optimize=optimize, imsize=imsize, batch_size=batch_size, verbose=verbose)
    else:
        sys.stderr.write(f"Error: Unkown dataset {ds}!\n")
        return

