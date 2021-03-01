import numpy as np
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


from utils import EMACallback, process_directory, ShowCallback, process_numpy
from layers import AugmentLayer
from hrgan import HrGAN
from pergan import PerGAN
from dcgan import DCGAN

# anime_path = "E:/project/data/process/picture/seeprettyface_anime_face/anime_face"
# anime_dataset = process_directory(anime_path, batch_size=16)

anime_path = "E:/project/data/process/picture/anime64.npy"
anime_dataset = process_numpy(np.load(anime_path), batch_size=64)

ema = EMACallback()
show = ShowCallback()
# model = DCGAN(filter_num=16)
model = PerGAN(filter_num=16)
model.build((64, 64, 3))
model.compile(d_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5, beta_1=0.5, beta_2=0.9),
              g_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5, beta_1=0.5, beta_2=0.9),
              loss=keras.losses.MeanSquaredError())
model.summary()
keras.utils.plot_model(model.gen, show_shapes=True, to_file="gen.png")
keras.utils.plot_model(model.dis, show_shapes=True, to_file="dis.png")
model.fit(anime_dataset, epochs=20, callbacks=[ema, show])
