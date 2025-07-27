#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
env_check.py
- VM/서버에서 TensorFlow GPU 학습 가능성 점검 스크립트
- Python 3.9 가상환경에서 실행 권장
"""

import os
import sys
import json
import time
import platform
import shutil
import subprocess
from datetime import datetime
from typing import Tuple, Optional

MIN_DRIVER_LINUX = (525, 60, 13)   # TF 문서 기준
MIN_DRIVER_WSL   = (528, 33, 0)

def print_header():
    print("="*80)
    print("TensorFlow GPU Environment Check")
    print("="*80)
    print(f"Timestamp          : {datetime.now().isoformat()}")
    print(f"Python executable  : {sys.executable}")
    print(f"Python version     : {sys.version.splitlines()[0]}")
    print(f"Platform           : {platform.platform()}")
    print()

def run(cmd: str, timeout: int = 20) -> Tuple[int, str]:
    try:
        completed = subprocess.run(
            cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout
        )
        return completed.returncode, completed.stdout.strip()
    except Exception as e:
        return 1, f"<error: {e}>"

def parse_version_tuple(s: str) -> Tuple[int, ...]:
    nums = []
    for part in s.split("."):
        try:
            nums.append(int(part))
        except ValueError:
            break
    return tuple(nums) if nums else (0,)

def cmp_version(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    la, lb = len(a), len(b)
    if la < lb:
        a = a + (0,) * (lb - la)
    elif lb < la:
        b = b + (0,) * (la - lb)
    return (a > b) - (a < b)

def check_os():
    print("## OS / Kernel")
    code, out = run("cat /etc/os-release")
    print("$ cat /etc/os-release")
    print(out, "\n")
    code, out = run("uname -a")
    print("$ uname -a")
    print(out, "\n")
    code, out = run("hostnamectl")
    print("$ hostnamectl")
    print(out, "\n")

def check_cpu_mem_disk():
    print("## CPU / Memory / Disk")
    for cmd in [
        "lscpu",
        "nproc",
        "free -h",
        "df -h",
    ]:
        print(f"$ {cmd}")
        _, out = run(cmd)
        print(out, "\n")

def check_gpu_driver():
    print("## GPU / Driver / Compute Capability")
    have_nvidia_smi = shutil.which("nvidia-smi") is not None
    driver_version = None
    compute_caps = []

    if have_nvidia_smi:
        cmd = "nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv"
        print(f"$ {cmd}")
        code, out = run(cmd)
        print(out, "\n")

        # 파싱
        lines = out.splitlines()
        if len(lines) >= 2:
            for i, line in enumerate(lines[1:], start=1):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    name, drv, cc = parts[0], parts[1], parts[2]
                    if driver_version is None:
                        driver_version = drv
                    compute_caps.append(cc)
        else:
            # 구형 드라이버일 경우 compute_cap 열이 없을 수 있음
            code, out = run("nvidia-smi")
            print("$ nvidia-smi")
            print(out, "\n")

    else:
        print("$ nvidia-smi")
        print("<nvidia-smi not found>\n")

    print("$ lspci | grep -i -E 'nvidia|vga'")
    _, out = run("lspci | grep -i -E 'nvidia|vga'")
    print(out, "\n")

    print("$ nvcc --version")
    _, out = run("nvcc --version")
    print(out, "\n")

    return driver_version, compute_caps

def list_python_packages():
    print("## Python packages (key)")
    # importlib.metadata 로 특정 패키지 버전 빠르게 표시
    try:
        import importlib.metadata as md
    except Exception:
        import importlib_metadata as md  # type: ignore

    pkgs = {}
    def v(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "<not installed>"

    targets = [
        "tensorflow", "keras", "numpy", "h5py",
        "nvidia-cublas-cu12", "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12",
        "nvidia-cusolver-cu12", "nvidia-cusparse-cu12", "nvidia-nccl-cu12",
        "nvidia-cuda-toolkit", "nvidia-cuda-runtime", "nvidia-cublas", "nvidia-cudnn",
        "nvidia-nvjitlink-cu12",
    ]
    for name in targets:
        pkgs[name] = v(name)
    print(json.dumps(pkgs, indent=2), "\n")

    print("$ pip show tensorflow")
    _, out = run("pip show tensorflow")
    print(out, "\n")

def check_tf():
    print("## TensorFlow build info / GPU detection")
    tf_ok = False
    try:
        import tensorflow as tf
        print("TensorFlow version:", tf.__version__)
        print("Built with CUDA:", tf.test.is_built_with_cuda())

        # Build info
        try:
            info = tf.sysconfig.get_build_info()
        except Exception:
            info = {}
        keys = ("cuda_version", "cudnn_version", "is_cuda_build")
        filtered = {k: info.get(k) for k in keys}
        print("Build info:", json.dumps(filtered, indent=2))

        # Device list
        gpus = tf.config.list_physical_devices('GPU')
        print("Physical GPU devices:", gpus)

        # Mixed precision policy (if available)
        try:
            from tensorflow.keras import mixed_precision
            print("Mixed precision global policy:", mixed_precision.global_policy())
        except Exception as e:
            print("Mixed precision check error:", e)

        tf_ok = True

        # Quick GPU test
        if gpus:
            print("\n-- GPU quick matmul test --")
            try:
                with tf.device('/GPU:0'):
                    a = tf.random.normal([4096, 4096])
                    b = tf.random.normal([4096, 4096])

                    # warmup
                    for _ in range(3):
                        _ = tf.matmul(a, b)

                    # sync helper
                    def sync():
                        try:
                            tf.experimental.sync_devices()
                        except Exception:
                            # Fallback: force materialization
                            _ = tf.reduce_sum(a).numpy()

                    sync()
                    t0 = time.time()
                    for _ in range(10):
                        _ = tf.matmul(a, b)
                    sync()
                    dt = time.time() - t0
                    print(f"10 matmuls on GPU: {dt:.3f} sec")
            except Exception as e:
                print("GPU matmul test error:", repr(e))

    except Exception as e:
        print("TensorFlow import error:", repr(e))

    return tf_ok

def print_env():
    print("## Environment variables (subset)")
    keys = [
        "VIRTUAL_ENV", "CONDA_PREFIX",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PATH",
        "TF_CPP_MIN_LOG_LEVEL",
    ]
    env = {k: os.environ.get(k, "") for k in keys}
    print(json.dumps(env, indent=2), "\n")

def summarize(driver_version: Optional[str], compute_caps, tf_ok: bool):
    print("\n" + "="*80)
    print("SUMMARY / Heuristic judgement")
    print("="*80)

    # 드라이버 기준 판단
    drv_tuple = parse_version_tuple(driver_version or "0.0.0")
    is_wsl = "microsoft" in platform.release().lower() or "wsl" in platform.platform().lower()
    min_drv = MIN_DRIVER_WSL if is_wsl else MIN_DRIVER_LINUX
    drv_ok = cmp_version(drv_tuple, min_drv) >= 0

    print(f"- Detected driver version : {driver_version or '<unknown>'}")
    print(f"- Minimum required driver : {'.'.join(map(str,min_drv))} ({'WSL' if is_wsl else 'Linux'})")
    print(f"- Driver OK               : {drv_ok}")

    # compute capability
    cc_ok = None
    if compute_caps:
        # 전체가 3.5 이상인지
        def cc_tuple(s):
            try:
                major, minor = s.split(".")
                return int(major), int(minor)
            except Exception:
                return (0, 0)
        caps = [cc_tuple(s) for s in compute_caps]
        cc_ok = all((maj > 3) or (maj == 3 and minr >= 5) for maj, minr in caps)
        print(f"- Compute capabilities    : {compute_caps}")
        print(f"- Compute capability OK   : {cc_ok}")
    else:
        print("- Compute capabilities    : <unknown> (old driver or query failure)")

    print(f"- TensorFlow import OK    : {tf_ok}")

    ready = bool(drv_ok and (cc_ok is not False) and tf_ok)
    print(f"\n>>> OVERALL GPU READY     : {ready}")
    if not ready:
        print("\nHints:")
        print("  * 드라이버 버전이 낮으면 업데이트 필요.")
        print("  * 가상환경에서 `pip install 'tensorflow[and-cuda]==2.17.0'` 재설치 권장.")
        print("  * GPU가 비어있으면 TF 문서의 심볼릭 링크 보정 절차를 수행:")
        print("      pushd $(dirname $(python -c 'print(__import__(\"tensorflow\").__file__)'))")
        print("      ln -svf ../nvidia/*/lib/*.so* .")
        print("      popd")
        print("      ln -sf $(find $(dirname $(dirname $(python -c \"import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)\"))/*/bin/) -name ptxas -print -quit) $VIRTUAL_ENV/bin/ptxas")
        print("  * 이후 `python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'` 재확인.")

def main():
    print_header()
    check_os()
    check_cpu_mem_disk()
    driver_version, compute_caps = check_gpu_driver()
    list_python_packages()
    tf_ok = check_tf()
    print_env()
    summarize(driver_version, compute_caps, tf_ok)

if __name__ == "__main__":
    main()
