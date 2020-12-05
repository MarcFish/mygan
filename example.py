import numpy as np
from PIL import Image
from mygan.utils import process_numpy
from mygan.models import CycleGAN


human_path = "./human64.npy"
anime_path = "./anime64.npy"
human_imgs = np.load(human_path)
anime_imgs = np.load(anime_path)
img_shape = (64, 64, 3)
human_dataset = process_numpy(human_imgs, batch_size=32)
anime_dataset = process_numpy(anime_imgs, batch_size=32)
model = CycleGAN(img_shape, batch_size=32)
# model.s_sample = list(human_dataset.take(1).as_numpy_iterator())[0]
# model.t_sample = list(anime_dataset.take(1).as_numpy_iterator())[0]
model.train(human_dataset, anime_dataset)
model.generate_samples(path="./results/cyclegan.png")
