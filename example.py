import numpy as np

from mygan.utils import process_numpy
from mygan.models import SpectralGAN

path = "./human64.npy"
imgs = np.load(path)
img_shape = (64, 64, 3)
dataset = process_numpy(imgs, batch_size=32)
model = SpectralGAN(512, img_shape, batch_size=32)
model.train(dataset)
model.generate_samples(path="./results/dcgan.png")
