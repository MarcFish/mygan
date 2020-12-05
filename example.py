import numpy as np
from PIL import Image
from mygan.utils import process_numpy
from mygan.models import CycleGAN


human_path = "./human64.npy"
anime_path = "./anime64.npy"
human_imgs = np.load(human_path)
anime_imgs = np.load(anime_path)
img_shape = (64, 64, 3)
human_dataset = process_numpy(human_imgs, batch_size=8)
anime_dataset = process_numpy(anime_imgs, batch_size=8)
model = CycleGAN(img_shape, batch_size=8)
# model.train(human_dataset, anime_dataset)
model.s_sample = list(human_dataset.take(1).as_numpy_iterator())[0]
model.t_sample = list(human_dataset.take(1).as_numpy_iterator())[0]
model.generate_samples(path="./results/cyclegan.png")
