import numpy as np
from PIL import Image
from mygan.utils import process_numpy
from mygan.models import DCGAN

anime_path = "./anime64.npy"
anime_imgs = np.load(anime_path)
img_shape = (64, 64, 3)
anime_dataset = process_numpy(anime_imgs, batch_size=32)
model = DCGAN(512, img_shape, batch_size=32, perform_gp=True)
model.train(anime_dataset)
model.generate_samples(path="./results/dcgangp.png")
