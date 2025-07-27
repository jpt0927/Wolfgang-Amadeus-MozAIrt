# gpu_benchmark.py
import time, json, tensorflow as tf

print("TF:", tf.__version__)
info = tf.sysconfig.get_build_info()
print("Build info:", json.dumps(
    {k: info.get(k) for k in ("cuda_version","cudnn_version","is_cuda_build")},
    indent=2))

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

# 간단 matmul 벤치마크
def bench(device):
    with tf.device(device):
        a = tf.random.normal([4096,4096])
        b = tf.random.normal([4096,4096])
        # 워밍업
        for _ in range(3):
            _ = tf.matmul(a,b)
        # 동기화
        try: tf.experimental.sync_devices()
        except: _ = tf.reduce_sum(a).numpy()
        t0 = time.time()
        for _ in range(10):
            _ = tf.matmul(a,b)
        try: tf.experimental.sync_devices()
        except: _ = tf.reduce_sum(a).numpy()
        return time.time() - t0

if gpus:
    dt_gpu = bench("/GPU:0")
    print(f"GPU time (10 matmuls): {dt_gpu:.3f} s")
else:
    print("No GPU detected.")

# CPU 비교(선택)
dt_cpu = bench("/CPU:0")
print(f"CPU time (10 matmuls): {dt_cpu:.3f} s")
