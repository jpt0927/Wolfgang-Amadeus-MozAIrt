import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras import mixed_precision
from pathlib import Path

# ======================================
# 전역 설정
# ======================================
mixed_precision.set_global_policy("mixed_float16")
ROOT_DIR = Path(__file__).resolve().parents[1]

# 하이퍼파라미터
TIME_RESOLUTION = 0.05
LATENT_DIM = 32
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
SEQUENCE_LENGTH = 200

# 데이터셋/출력 경로 태그
DATASET_TAG = "Country"
DATA_DIR = ROOT_DIR / "data" / "processed" / DATASET_TAG
OUTPUT_DIR = ROOT_DIR / "outputs" / DATASET_TAG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / f"vae_{DATASET_TAG}.weights.h5"
VAE_SAVE_PATH   = OUTPUT_DIR / f"vae_{DATASET_TAG}.h5"
ENC_SAVE_PATH   = OUTPUT_DIR / f"encoder_{DATASET_TAG}.h5"
DEC_SAVE_PATH   = OUTPUT_DIR / f"decoder_{DATASET_TAG}.h5"

# ======================================
# GPU 설정
# ======================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ======================================
# 데이터 로딩 (슬라이딩 윈도우)
# ======================================
def load_data(input_dir=DATA_DIR, seq_length=SEQUENCE_LENGTH, step=100):
    """
    .npz 파일의 'roll' 배열(T,128)을 읽어 (N, seq_length, 128) 시퀀스 생성.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files in {input_dir}")

    sequences = []
    for fp in files:
        try:
            with np.load(fp) as npz_file:
                if "roll" not in npz_file:
                    print(f"[WARN] 'roll' key not found in {fp.name}, skip.")
                    continue
                roll = npz_file["roll"].astype(np.float32)

                if roll.ndim != 2 or roll.shape[1] != 128:
                    print(f"[WARN] Unexpected roll shape {roll.shape} in {fp.name}, skip.")
                    continue

                T = roll.shape[0]
                if T < seq_length:
                    continue

                for i in range(0, T - seq_length + 1, step):
                    sequences.append(roll[i:i + seq_length])

        except Exception as e:
            print(f"[WARN] Error reading {fp.name}: {e}")

    if not sequences:
        raise RuntimeError(
            f"No sequences produced. Check seq_length={seq_length}, step={step}, and data in {input_dir}"
        )

    return np.stack(sequences, dtype=np.float32)

# ======================================
# VAE 구성요소
# ======================================
class Sampling(layers.Layer):
    """Reparameterization + KL(add_loss)"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        # 손실 안정성을 위해 float32에서 계산
        zm32 = tf.cast(z_mean, tf.float32)
        zv32 = tf.cast(z_log_var, tf.float32)
        eps  = tf.random.normal(tf.shape(zm32), dtype=tf.float32)
        z32  = zm32 + tf.exp(0.5 * zv32) * eps

        # KL loss 추가
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1.0 + zv32 - tf.square(zm32) - tf.exp(zv32), axis=-1)
        )
        self.add_loss(kl)

        # 정책 dtype으로 캐스팅해 반환
        return tf.cast(z32, z_mean.dtype)

def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def build_decoder(latent_dim, output_shape):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64)(latent_inputs)
    seq_len = output_shape[0]
    x = layers.RepeatVector(seq_len)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    # 출력은 float32로 캐스팅(혼합정밀도 안정성)
    outputs = layers.TimeDistributed(
        layers.Dense(output_shape[1], activation="sigmoid", dtype="float32")
    )(x)
    decoder = models.Model(latent_inputs, outputs, name="decoder")
    return decoder

def reconstruction_loss_fn(y_true, y_pred):
    # 손실은 float32에서 계산
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))

def build_vae(input_shape, latent_dim):
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape)

    inputs = layers.Input(shape=input_shape)
    _, _, z = encoder(inputs)
    reconstructed = decoder(z)

    vae = models.Model(inputs, reconstructed, name="vae")
    vae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=reconstruction_loss_fn,        # KL은 Sampling 레이어에서 add_loss로 추가됨
        jit_compile=True
    )
    return vae, encoder, decoder

# ======================================
# 데이터 준비 & 모델 생성
# ======================================
data = load_data(seq_length=SEQUENCE_LENGTH)
input_shape = (SEQUENCE_LENGTH, 128)
vae, encoder, decoder = build_vae(input_shape, LATENT_DIM)

# ======================================
# 콜백
# ======================================
def generate_sample(epoch, logs):
    if epoch % 10 == 0:
        z = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
        generated_roll = decoder.predict(z, verbose=0)
        # 필요 시 MIDI 저장 구현
        print(f"[INFO] Generated sample at epoch {epoch} | shape={generated_roll.shape}")

generate_sample_callback = LambdaCallback(on_epoch_end=generate_sample)

checkpoint_callback = ModelCheckpoint(
    str(CHECKPOINT_PATH),
    save_best_only=True,
    save_weights_only=True,
    monitor="loss", mode="min"
)

# 재시작 시 체크포인트 로드
if CHECKPOINT_PATH.exists():
    print(f"[INFO] Load checkpoint weights: {CHECKPOINT_PATH}")
    vae.load_weights(str(CHECKPOINT_PATH))

# ======================================
# 학습
# ======================================
if __name__ == "__main__":
    vae.fit(
        data, data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_callback, generate_sample_callback],
        verbose=1
    )

    vae.save(str(VAE_SAVE_PATH))
    encoder.save(str(ENC_SAVE_PATH))
    decoder.save(str(DEC_SAVE_PATH))

    print("[SAVED]")
    print("  VAE     :", VAE_SAVE_PATH)
    print("  Encoder :", ENC_SAVE_PATH)
    print("  Decoder :", DEC_SAVE_PATH)

    # 샘플 생성
    def generate_music():
        z = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
        return decoder.predict(z, verbose=0)

    gen = generate_music()
    print("sample shape/min/max:",
          gen.shape, float(gen.min()), float(gen.max()))
