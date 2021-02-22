import numpy as np
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


from utils import EMACallback, process_numpy, ShowCallback
from layers import AugmentLayer
from dcgan import DCGAN

k = 64
anime_path = f"./cat{k}.npy"
anime_imgs = np.load(anime_path)
anime_dataset = process_numpy(anime_imgs, batch_size=32)

ema = EMACallback()
show = ShowCallback()
model = DCGAN(filter_num=32)
model.build(anime_imgs.shape[1:])
model.compile(d_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
              g_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
              loss=keras.losses.BinaryCrossentropy(from_logits=True))

model.fit(anime_dataset, epochs=10, callbacks=[ema, show])
