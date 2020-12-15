import numpy as np
from PIL import Image
from mygan.utils import process_numpy
from mygan.models import DCGAN, RaGAN

anime_path = "./cat128.npy"
anime_imgs = np.load(anime_path)
img_shape = (128, 128, 3)
anime_dataset = process_numpy(anime_imgs, batch_size=8)
model = RaGAN(100, img_shape, batch_size=8, perform_gp=True, filter_num=8)
model.train(anime_dataset)
# model.generate_samples(path="./results/dcgangp.png")
