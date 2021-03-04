import numpy as np
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


from utils import EMACallback, process_directory, ShowCallback, process_numpy, StyleShowCallback, PathCallback
from hrgan import HrGAN
from pergan import PerGAN
from dcgan import DCGAN
from stylegan2 import StyleGAN2
from stylegan import StyleGAN

# anime_path = "E:/project/data/process/picture/seeprettyface_anime_face/anime_face"
# anime_dataset = process_directory(anime_path, batch_size=16)

anime_path = "E:/project/data/process/picture/anime64.npy"
anime_dataset = process_numpy(np.load(anime_path), batch_size=32)

ema = EMACallback()
# show = ShowCallback()
show = StyleShowCallback()
path = PathCallback()

model = HrGAN(latent_dim=512, filter_num=4)
model.build((64, 64, 3))
model.compile(d_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5, beta_1=0., beta_2=0.99),
              g_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5, beta_1=0., beta_2=0.99),
              loss=keras.losses.MeanSquaredError())
model.summary()
keras.utils.plot_model(model.gen, show_shapes=True, to_file="gen.png", expand_nested=True)
keras.utils.plot_model(model.dis, show_shapes=True, to_file="dis.png", expand_nested=True)
model.fit(anime_dataset, epochs=50, callbacks=[ema, show, path])
