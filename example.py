import numpy as np
from PIL import Image
from mygan.utils import process_numpy
from mygan.models import StyleGAN


human_path = "./human64.npy"
human_imgs = np.load(human_path)
img_shape = (64, 64, 3)
human_dataset = process_numpy(human_imgs, batch_size=8)
model = StyleGAN(100, img_shape, batch_size=8)
model.train(human_dataset)
model.generate_samples(path="./results/stylegan.png")
