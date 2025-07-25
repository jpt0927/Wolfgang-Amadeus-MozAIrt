# 모델에 먹일 데이터 전처리. 모델이 읽기 편한 악보 만듦

# 커스터마이징 포인트
# 시간 단위            TIME_RESOLUTION = 0.1 (100ms) 등
# velocity 처리		  velocity / 127.0로 정규화 가능
# 길이 자르기       	MAX_DURATION 변경


import os
import json
import numpy as np

INPUT_DIR = "data/extracted/Classical"
OUTPUT_DIR = "data/processed/Classical"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 전처리 하이퍼파라미터
TIME_RESOLUTION = 0.05  # 50ms 단위 (20 steps/sec)
MAX_DURATION = 30       # 30초까지만 사용
MAX_STEPS = int(MAX_DURATION / TIME_RESOLUTION)

def notes_to_piano_roll(notes):
    roll = np.zeros((MAX_STEPS, 128), dtype=np.float32)
    
    for note in notes:
        time_sec = note["time"]
        pitch = note["pitch"]
        velocity = note["velocity"]
        
        step = int(time_sec / TIME_RESOLUTION)
        if 0 <= step < MAX_STEPS and 0 <= pitch < 128:
            roll[step][pitch] = 1.0  # 지금은 모든 음의 세기를 1.0으로 고정함. 세기 정보를 반영하면 음의 세기 학습 가능. velocity / 127.0 (정규화)
    
    return roll

def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    for fname in files:
        full_path = os.path.join(INPUT_DIR, fname)
        print(f"🎼 Processing: {fname}")
        try:
            with open(full_path, "r") as f:
                notes = json.load(f)
            roll = notes_to_piano_roll(notes)
            
            save_path = os.path.join(OUTPUT_DIR, fname.replace(".json", ".npy"))
            np.save(save_path, roll)
            print(f"✅ Saved to: {save_path}")
        except Exception as e:
            print(f"❌ Failed on {fname}: {e}")

if __name__ == "__main__":
    main()
