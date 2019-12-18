# GANs

A CLI for training generative adversarial networks on prespecified
artisitic datasets and displaying the results. Written during an
exploratory research project during the fall of 2019 with Adam Davenport.

## Usage

The main program is `gans.py`, and has two subcommands. Run `python3
gans.py train -h` and `python3 gans.py results -h` to see the possible
options. Below is an example of running a training session:
```./gans.py train --image-size 64 --nfeatures 32 --learning-rate 0.0001 --batch-size 64 --iteration 50000 --sample-interval 200 --gp-enabled true ross ross-3```

## Visualizing Results

One possible way of visualizing the results of the training session is
via a GIF, such as the ones below. They show the generator's output
for the same latent vectors at regular intervals during the training
session.

![ross-2-static-images](figures/ross-2-static-images.gif)
![impressionism-1-static-images](figures/impressionism-1-static-images.gif)
![cubism-wgan-1-static-images](figures/cubism-wgan-1-static-images.gif)

