import numpy as np
from PIL import Image
from mygan.utils import process_numpy
from mygan.models import GAN, DCGAN, RaGAN, FastGAN


k=64
anime_path = f"./cat{k}.npy"
anime_imgs = np.load(anime_path)
img_shape = (k, k, 3)
anime_dataset = process_numpy(anime_imgs, batch_size=32)
model = DCGAN(latent_dim=100, img_shape=img_shape, batch_size=32, filter_num=32)
model.train(anime_dataset)
# model.generate_samples(path="./results/dcgangp.png")
