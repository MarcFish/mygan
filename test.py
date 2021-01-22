import numpy as np
from PIL import Image
from mygan.utils import process_numpy
from mygan.models import FastGAN


img_shape = (1024, 1024, 3)
model = FastGAN(100, img_shape, batch_size=32, filter_num=32)

