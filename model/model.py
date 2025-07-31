# model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import mixed_precision
from pathlib import Path
import pretty_midi

# ======================================
# 전역 설정 및 하이퍼파라미터
# ======================================
mixed_precision.set_global_policy("mixed_float16")
try:
    ROOT_DIR = Path(__file__).resolve().parents[1]
except NameError:
    ROOT_DIR = Path.cwd()

MAX_SEQUENCE_LENGTH = 500
TIME_RESOLUTION = 0.05
LATENT_DIM = 128
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 1e-5

DATASET_TAG = "Classical"
DATA_DIR = ROOT_DIR / "data" / "raw" / DATASET_TAG
OUTPUT_DIR = ROOT_DIR / "outputs" / f"{DATASET_TAG}_CNN_Padding_Midi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / f"vae_{DATASET_TAG}_E{EPOCHS}.weights.h5"
VAE_SAVE_PATH   = OUTPUT_DIR / f"vae_{DATASET_TAG}_E{EPOCHS}.keras"
ENC_SAVE_PATH   = OUTPUT_DIR / f"encoder_{DATASET_TAG}_E{EPOCHS}.keras"
DEC_SAVE_PATH   = OUTPUT_DIR / f"decoder_{DATASET_TAG}_E{EPOCHS}.keras"

# ======================================
# 함수 정의
# ======================================
def set_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

class Sampling(layers.Layer):
    """Reparameterization trick."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        zm32, zv32 = tf.cast(z_mean, tf.float32), tf.cast(z_log_var, tf.float32)
        eps = tf.random.normal(tf.shape(zm32))
        z32 = zm32 + tf.exp(0.5 * zv32) * eps
        return tf.cast(z32, z_mean.dtype)

class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        # ✅ 이 부분이 핵심 수정 사항입니다.
        x, y = data # 데이터를 입력(x)과 정답(y)으로 분리
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x) # 인코더에는 입력(x)만 전달
            reconstruction = self.decoder(z)
            
            y_true = tf.cast(y, tf.float32)
            y_pred = tf.cast(reconstruction, tf.float32)
            epsilon = 1e-7
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(bce, axis=[1, 2]))
            
            z_mean_f32 = tf.cast(z_mean, tf.float32)
            z_log_var_f32 = tf.cast(z_log_var, tf.float32)
            kl_loss = -0.5 * (1 + z_log_var_f32 - tf.square(z_mean_f32) - tf.exp(z_log_var_f32))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {m.name: m.result() for m in self.metrics}

def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", strides=(2, 2))(x)
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.shape_before_flatten = shape_before_flatten 
    return encoder

def build_decoder(latent_dim, shape_before_flatten):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(np.prod(shape_before_flatten), activation="relu")(latent_inputs)
    x = layers.Reshape(shape_before_flatten)(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same", strides=(2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same", strides=(2, 2))(x)
    outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", dtype="float32")(x)
    decoder = models.Model(latent_inputs, outputs, name="decoder")
    return decoder

def build_and_compile_vae(encoder, decoder):
    vae = VAE(encoder, decoder)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    vae.compile(optimizer=optimizer)
    return vae

checkpoint_callback = ModelCheckpoint(
    str(CHECKPOINT_PATH), save_best_only=True,
    save_weights_only=True, monitor="total_loss", mode="min"
)