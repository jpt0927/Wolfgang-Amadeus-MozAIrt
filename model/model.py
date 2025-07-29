# model.py
import os
import json
import random
from typing import Optional, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LambdaCallback,
    Callback,
    EarlyStopping,
)
from tensorflow.keras import mixed_precision
from pathlib import Path

# =========================================================
# 전역 설정 & 재현성
# =========================================================
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

mixed_precision.set_global_policy("mixed_float16")
ROOT_DIR = Path(__file__).resolve().parents[1]

# =========================================================
# 하이퍼파라미터
# =========================================================
LATENT_DIM       = 32
BATCH_SIZE       = 64
EPOCHS           = 20
LEARNING_RATE    = 3e-4
SEQUENCE_LENGTH  = 128
STEP             = 10

# KL
KL_MAX_WEIGHT      = 0.2
KL_WARMUP_EPOCHS   = 5
FREE_NATS_PER_DIM  = 0.05

# 데이터 필터링
MIN_POS_RATIO = 5e-5

# 훈련용 규제
INIT_PROB        = 0.25
TARGET_ACTIVITY  = 0.05
LAMBDA_ACTIVITY  = 0.10
R50_MIN          = 0.02
LAMBDA_R50       = 0.50
TAU_R50          = 0.10

# 추론 임계값 정책
DEFAULT_TARGET_RATE = float(os.getenv("TARGET_RATE", 0.03))
MIN_THR = 0.01
MAX_THR = 0.25

# =========================================================
# 경로
# =========================================================
DATASET_TAG  = "Classical"
DATA_DIR     = ROOT_DIR / "data" / "processed" / DATASET_TAG
OUTPUT_DIR   = ROOT_DIR / "outputs" / DATASET_TAG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH        = OUTPUT_DIR / f"vae_{DATASET_TAG}_E{EPOCHS}.weights.h5"
VAE_SAVE_PATH          = OUTPUT_DIR / f"vae_{DATASET_TAG}_E{EPOCHS}.keras"
ENC_SAVE_PATH          = OUTPUT_DIR / f"encoder_{DATASET_TAG}_E{EPOCHS}.keras"
DEC_SAVE_PATH          = OUTPUT_DIR / f"decoder_{DATASET_TAG}_E{EPOCHS}.keras"
DEC_LOGITS_SAVE_PATH   = OUTPUT_DIR / f"decoder_logits_{DATASET_TAG}_E{EPOCHS}.keras"
DEC_CALIB_SAVE_PATH    = OUTPUT_DIR / f"decoder_calibrated_{DATASET_TAG}_E{EPOCHS}.keras"
DEC_CALIB_META_JSON    = OUTPUT_DIR / f"decoder_calibrated_{DATASET_TAG}_E{EPOCHS}.meta.json"

# =========================================================
# GPU 설정
# =========================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

# =========================================================
# 유틸
# =========================================================
def prob_to_logit(p: float) -> float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))

def decision_thresholds(
    probs: np.ndarray,
    pos_weight: float,
    target_rate: float = DEFAULT_TARGET_RATE,
    min_thr: float = MIN_THR,
    max_thr: float = MAX_THR,
):
    flat = probs.ravel()
    t_star = 1.0 / (1.0 + float(pos_weight))
    q = max(0.0, 1.0 - float(target_rate))
    t_quant = float(np.quantile(flat, q)) if flat.size else t_star
    t_on = float(np.clip(max(t_star, t_quant), min_thr, max_thr))
    t_off = float(max(min_thr * 0.5, t_on * 0.5))
    return t_on, t_off, t_star, t_quant

def apply_hysteresis(probs: np.ndarray, t_on: float, t_off: float):
    T, P = probs.shape
    out = np.zeros((T, P), dtype=bool)
    state = np.zeros(P, dtype=bool)
    for t in range(T):
        p = probs[t]
        turn_on  = (~state) & (p >= t_on)
        turn_off = state & (p <  t_off)
        state = (state | turn_on) & (~turn_off)
        out[t] = state
    return out

def continuous_on_rate(g: np.ndarray, thr=0.5, span=4) -> float:
    L = g.shape[0]
    if L < span:
        return 0.0
    mask = g[0 : L - span + 1] > thr
    for t in range(1, span):
        mask &= g[t : L - span + 1 + t] > thr
    return float(mask.mean())

# ---- 바이어스 보정: 가장 단순 정책 ----
def choose_t_on(pos_weight: float,
                target_rate: float = DEFAULT_TARGET_RATE,
                t_min: float = MIN_THR,
                t_max: float = MAX_THR) -> float:
    # 보수적: t_on = clip(max(t*, target_rate))
    t_star = 1.0 / (1.0 + float(pos_weight))
    t_on = max(t_star, float(target_rate))
    return float(np.clip(t_on, t_min, t_max))

def export_calibrated_from_logits(decoder_logits: tf.keras.Model,
                                  latent_dim: int,
                                  t_on: float,
                                  save_path: Path) -> tf.keras.Model:
    # b = -logit(t_on); logits+b -> sigmoid. 0.5 임계 == t_on 임계와 동등
    t_on = float(np.clip(t_on, 1e-6, 1.0 - 1e-6))
    b = -prob_to_logit(t_on)

    z_in = layers.Input(shape=(latent_dim,))
    logits = decoder_logits(z_in)
    shifted = layers.Lambda(lambda x: tf.cast(x, tf.float32) + b, dtype="float32")(logits)
    probs = layers.Activation("sigmoid", dtype="float32")(shifted)
    dec_cal = models.Model(z_in, probs, name="decoder_calibrated")
    dec_cal.save(str(save_path))
    print(f"[SAVED] decoder_calibrated → {save_path} (bias={b:.6f}, t_on={t_on:.6f})")
    return dec_cal

# =========================================================
# 데이터 로딩
# =========================================================
def load_data(input_dir=DATA_DIR, seq_length=SEQUENCE_LENGTH, step=STEP):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files in {input_dir}")

    sequences = []
    kept, skipped_shape, skipped_sparse = 0, 0, 0
    for fp in files:
        try:
            with np.load(fp) as npz_file:
                if "roll" not in npz_file:
                    print(f"[WARN] 'roll' key not found in {fp.name}, skip.")
                    continue
                roll = npz_file["roll"].astype(np.float32)
                if roll.ndim != 2 or roll.shape[1] != 128:
                    skipped_shape += 1
                    continue
                T = roll.shape[0]
                if T < seq_length:
                    continue
                for i in range(0, T - seq_length + 1, step):
                    seq = roll[i : i + seq_length]
                    if MIN_POS_RATIO is not None and (seq > 0).mean() < MIN_POS_RATIO:
                        skipped_sparse += 1
                        continue
                    sequences.append(seq); kept += 1
        except Exception as e:
            print(f"[WARN] Error reading {fp.name}: {e}")

    if not sequences:
        raise RuntimeError("No sequences produced. Check params & data.")
    print(f"[INFO] sequences kept={kept}, skipped_shape={skipped_shape}, skipped_sparse={skipped_sparse}")
    return np.stack(sequences, dtype=np.float32)

# =========================================================
# KL 스케줄러
# =========================================================
KL_WEIGHT = tf.Variable(0.0, trainable=False, dtype=tf.float32)

class KLWeightScheduler(Callback):
    def __init__(self, warmup_epochs=KL_WARMUP_EPOCHS, max_weight=KL_MAX_WEIGHT):
        super().__init__()
        self.warmup = max(1, int(warmup_epochs))
        self.maxw = float(max_weight)
    def on_epoch_begin(self, epoch, logs=None):
        w = self.maxw * min((epoch + 1) / self.warmup, 1.0)
        KL_WEIGHT.assign(w)
        if epoch == 0 or (epoch + 1) == self.warmup:
            print(f"[INFO] KL_WEIGHT set to {float(w):.4f} (epoch {epoch})")

# =========================================================
# 네트워크
# =========================================================
BIAS_INIT = prob_to_logit(INIT_PROB)

class Sampling(layers.Layer):
    def __init__(self, free_nats_per_dim=FREE_NATS_PER_DIM, **kwargs):
        super().__init__(**kwargs)
        self.free_nats_per_dim = float(free_nats_per_dim)
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss_tracker", dtype=tf.float32)
    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        zm32 = tf.cast(z_mean, tf.float32)
        zv32 = tf.cast(z_log_var, tf.float32)
        eps  = tf.random.normal(tf.shape(zm32), dtype=tf.float32)
        z32  = zm32 + tf.exp(0.5 * zv32) * eps
        kl_per_sample = -0.5 * tf.reduce_sum(1.0 + zv32 - tf.square(zm32) - tf.exp(zv32), axis=-1)
        latent_dim = tf.cast(tf.shape(zm32)[-1], tf.float32)
        free_nats = self.free_nats_per_dim * latent_dim
        kl_hinge = tf.nn.relu(kl_per_sample - free_nats)
        kl_loss = tf.reduce_mean(kl_hinge)
        self.add_loss(KL_WEIGHT * kl_loss)
        self.kl_tracker.update_state(kl_loss)
        return tf.cast(z32, z_mean.dtype)

def build_encoder(input_shape, latent_dim):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.LSTM(64)(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])
    enc = models.Model(inp, [z_mean, z_log_var, z], name="encoder")
    enc.sampling_layer = enc.layers[-1]
    return enc

def build_decoder(latent_dim, output_shape):
    latent_in = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64)(latent_in)
    x = layers.RepeatVector(output_shape[0])(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    logits = layers.TimeDistributed(
        layers.Dense(
            output_shape[1], activation=None, dtype="float32",
            bias_initializer=tf.keras.initializers.Constant(BIAS_INIT),
        )
    )(x)
    probs = layers.Activation("sigmoid", dtype="float32")(logits)
    dec = models.Model(latent_in, probs, name="decoder")
    dec.logits_model = models.Model(latent_in, logits, name="decoder_logits")
    return dec

# =========================================================
# 손실
# =========================================================
def weighted_bce_loss(pos_weight: float):
    pw = tf.constant(float(pos_weight), dtype=tf.float32)
    def loss(y_true, y_pred_logits):
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(y_pred_logits, tf.float32)
        xe = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=logits, pos_weight=pw)
        return tf.reduce_mean(xe)
    return loss

def smooth_r50_per_sample(probs, tau=TAU_R50):
    s = tf.sigmoid((tf.cast(probs, tf.float32) - 0.5) / tau)
    return tf.reduce_mean(s, axis=[1, 2])

def composite_loss(pos_weight, target_activity, lambda_activity, r50_min, lambda_r50):
    bce = weighted_bce_loss(pos_weight)
    ta = tf.constant(float(target_activity), dtype=tf.float32)
    lam_a = tf.constant(float(lambda_activity), dtype=tf.float32)
    rmin = tf.constant(float(r50_min), dtype=tf.float32)
    lam_r = tf.constant(float(lambda_r50), dtype=tf.float32)
    def loss(y_true, logits):
        l_bce = bce(y_true, logits)
        probs = tf.sigmoid(tf.cast(logits, tf.float32))
        activity = tf.reduce_mean(probs, axis=[1, 2])
        act_pen  = tf.reduce_mean(tf.square(activity - ta))
        r50_s = smooth_r50_per_sample(probs)
        r_pen = tf.nn.relu(rmin - r50_s)
        return l_bce + lam_a * act_pen + lam_r * tf.reduce_mean(r_pen)
    return loss

# =========================================================
# 데이터 & 모델 빌드
# =========================================================
data = load_data(seq_length=SEQUENCE_LENGTH, step=STEP)
p_rate = float(np.mean(data)) + 1e-12
print(f"[INFO] positive_rate={p_rate:.12f}")

raw_ratio = (1.0 - p_rate) / p_rate
pos_weight = float(max(1.0, raw_ratio ** 0.35))
print(f"[INFO] raw_pos_weight={raw_ratio:.2f}, effective_pos_weight={pos_weight:.2f}")

input_shape = (SEQUENCE_LENGTH, 128)
encoder = build_encoder(input_shape, LATENT_DIM)
decoder = build_decoder(LATENT_DIM, input_shape)

inp = layers.Input(shape=input_shape)
_, _, z = encoder(inp)
logits_out = decoder.logits_model(z)
vae = models.Model(inp, logits_out, name="vae_logits")

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
vae.compile(
    optimizer=optimizer,
    loss=composite_loss(
        pos_weight=pos_weight,
        target_activity=TARGET_ACTIVITY,
        lambda_activity=LAMBDA_ACTIVITY,
        r50_min=R50_MIN,
        lambda_r50=LAMBDA_R50,
    ),
    jit_compile=False,
)

# =========================================================
# 샘플 & 로깅
# =========================================================
def sample_and_log(epoch: int, target_rate: float = DEFAULT_TARGET_RATE):
    z = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
    logits = decoder.logits_model.predict(z, verbose=0)[0].astype(np.float32)
    probs  = 1.0 / (1.0 + np.exp(-logits))
    t_on, t_off, t_star, t_quant = decision_thresholds(
        probs, pos_weight=pos_weight, target_rate=target_rate
    )
    mask_hys = apply_hysteresis(probs, t_on, t_off)
    rates = {
        f">t*({t_star:.3f})": float((probs > t_star).mean()),
        f">q({t_quant:.3f})": float((probs > t_quant).mean()),
        "hys_on": float(mask_hys.mean()),
        ">0.50": float((probs > 0.50).mean()),
    }
    bin_rate_t  = continuous_on_rate(probs, thr=t_on,  span=4)
    bin_rate_05 = continuous_on_rate(probs, thr=0.50, span=4)
    print(
        f"[SAMPLE] ep {epoch:03d} | mean={probs.mean():.6f} std={probs.std():.6f} "
        f"rates={rates} | t_on={t_on:.4f} t_off={t_off:.4f} "
        f"| bin_rate(t_on)={bin_rate_t:.6f} bin_rate(0.5)={bin_rate_05:.6f}"
    )

def _on_epoch_end(epoch, logs):
    if epoch % 5 == 0:
        sample_and_log(epoch)

generate_sample_callback = LambdaCallback(on_epoch_end=_on_epoch_end)

# =========================================================
# 콜백
# =========================================================
checkpoint_callback = ModelCheckpoint(
    filepath=str(CHECKPOINT_PATH),
    save_best_only=True,
    save_weights_only=True,
    monitor="loss",
    mode="min",
)

es_callback = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    min_delta=0.001,
)

class KLLogger(Callback):
    def __init__(self, sampling_layer):
        super().__init__()
        self.sampling = sampling_layer
    def on_epoch_end(self, epoch, logs=None):
        kl_mean = float(self.sampling.kl_tracker.result().numpy())
        self.sampling.kl_tracker.reset_state()
        print(f"[KL] epoch {epoch:03d} | weight={float(KL_WEIGHT.numpy()):.4f} | kl_loss={kl_mean:.6f}")

kl_logger = KLLogger(encoder.sampling_layer)
kl_sched = KLWeightScheduler()

# =========================================================
# API 편의 (선택)
# =========================================================
def generate_from_noise(target_rate: float = DEFAULT_TARGET_RATE,
                        z: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if z is None:
        z = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
    logits = decoder.logits_model.predict(z, verbose=0)[0].astype(np.float32)
    probs  = 1.0 / (1.0 + np.exp(-logits))
    t_on, t_off, t_star, t_quant = decision_thresholds(
        probs, pos_weight=pos_weight, target_rate=target_rate
    )
    mask = apply_hysteresis(probs, t_on, t_off)
    return {
        "probs": probs,
        "mask": mask,
        "t_on": float(t_on),
        "t_off": float(t_off),
        "t_star": float(t_star),
        "t_quant": float(t_quant),
    }

# =========================================================
# 학습 & 저장 + 바이어스 보정 디코더 내보내기
# =========================================================
if __name__ == "__main__":
    print("모델 학습 시작...")
    history = vae.fit(
        data, data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.05,
        shuffle=True,
        callbacks=[checkpoint_callback, generate_sample_callback, kl_sched, es_callback, kl_logger],
        verbose=1,
    )

    # 원본 저장
    vae.save(str(VAE_SAVE_PATH))
    encoder.save(str(ENC_SAVE_PATH))
    decoder.save(str(DEC_SAVE_PATH))
    decoder.logits_model.save(str(DEC_LOGITS_SAVE_PATH))
    print("[SAVED]")
    print("  VAE-logits :", VAE_SAVE_PATH)
    print("  Encoder    :", ENC_SAVE_PATH)
    print("  Decoder    :", DEC_SAVE_PATH)
    print("  Dec-Logits :", DEC_LOGITS_SAVE_PATH)

    # 바이어스 보정 디코더 (간단 정책)
    t_on = choose_t_on(pos_weight=pos_weight, target_rate=DEFAULT_TARGET_RATE,
                       t_min=MIN_THR, t_max=MAX_THR)
    dec_calib = export_calibrated_from_logits(
        decoder_logits=decoder.logits_model,
        latent_dim=LATENT_DIM,
        t_on=t_on,
        save_path=DEC_CALIB_SAVE_PATH,
    )

    # 메타 저장
    meta = {
        "latent_dim": LATENT_DIM,
        "pos_weight": float(pos_weight),
        "target_rate_default": float(DEFAULT_TARGET_RATE),
        "t_on": float(t_on),
        "min_thr": float(MIN_THR),
        "max_thr": float(MAX_THR),
        "note": "decoder_calibrated는 0.5로 이진화하면 원본 디코더의 t_on 임계와 동등합니다.",
    }
    DEC_CALIB_META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"[SAVED] meta → {DEC_CALIB_META_JSON}")

    sample_and_log(EPOCHS)
