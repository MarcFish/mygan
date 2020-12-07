import numpy as np
from PIL import Image
from mygan.utils import process_numpy
from mygan.models import GAN

cat_path = "./cat64.npy"
cat_imgs = np.load(cat_path)
img_shape = (64, 64, 3)
cat_dataset = process_numpy(cat_imgs, batch_size=32)
model = GAN(512, img_shape, batch_size=32, perform_gp=True)
model.train(cat_dataset)
model.generate_samples(path="./results/gangp.png")
