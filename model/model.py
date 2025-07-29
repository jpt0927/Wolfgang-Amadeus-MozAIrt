# model.py - 변수 정의 순서 오류 수정 버전
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

# ✅ 모든 하이퍼파라미터를 함수 정의보다 먼저 선언합니다.
MAX_SEQUENCE_LENGTH = 500
TIME_RESOLUTION = 0.05
LATENT_DIM = 128
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-4

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

def load_data(input_dir=DATA_DIR, max_len=MAX_SEQUENCE_LENGTH, time_res=TIME_RESOLUTION):
    input_dir = Path(input_dir)
    if not input_dir.is_dir(): raise FileNotFoundError(f"Data directory not found: {input_dir}")
    files = list(input_dir.glob("*.mid*"))
    if not files: raise RuntimeError(f"No .mid or .midi files in {input_dir}")
    
    all_rolls = []
    fs = 1 / time_res
    
    for fp in files:
        try:
            midi_data = pretty_midi.PrettyMIDI(str(fp))
            roll = midi_data.get_piano_roll(fs=fs)
            binarized_roll = (roll.T > 0).astype(np.float32)
            all_rolls.append(binarized_roll)
        except Exception as e: 
            print(f"[WARN] Error processing {fp.name}: {e}")

    if not all_rolls: raise RuntimeError(f"No valid data loaded from {input_dir}")
    
    padded_rolls = pad_sequences(
        all_rolls, maxlen=max_len, padding='post', truncating='post', dtype='float32'
    )
    return np.expand_dims(padded_rolls, -1)

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        zm32, zv32 = tf.cast(z_mean, tf.float32), tf.cast(z_log_var, tf.float32)
        eps = tf.random.normal(tf.shape(zm32))
        z32 = zm32 + tf.exp(0.5 * zv32) * eps
        kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1.0 + zv32 - tf.square(zm32) - tf.exp(zv32), axis=-1))
        self.add_loss(kl)
        return tf.cast(z32, z_mean.dtype)

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

def reconstruction_loss_fn(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(tf.reduce_sum(bce, axis=[1, 2]))

def build_vae(encoder, decoder):
    inputs = encoder.input
    z_mean, z_log_var, z = encoder(inputs)
    reconstructed = decoder(z)
    vae = models.Model(inputs, reconstructed, name="vae")
    vae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=reconstruction_loss_fn,
        jit_compile=True
    )
    return vae

checkpoint_callback = ModelCheckpoint(
    str(CHECKPOINT_PATH), save_best_only=True,
    save_weights_only=True, monitor="loss", mode="min"
)