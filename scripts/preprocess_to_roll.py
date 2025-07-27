import os
import json
import numpy as np
import sys
import gc  # Garbage Collection 라이브러리

INPUT_DIR = "../data/extracted/Rock_Pop"
OUTPUT_DIR = "../data/processed/Rock_Pop"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_RESOLUTION = 0.05  # 50ms 단위 (20 steps/sec)
MAX_DURATION_LIMIT = 600  # 최대 600초로 제한

def get_max_duration(notes):
    """ notes의 최대 시간을 찾아 MAX_DURATION 반환 """
    return max(note['time'] for note in notes)

def notes_to_piano_roll(notes):
    MAX_DURATION = get_max_duration(notes)
    # MAX_DURATION을 600초로 제한
    MAX_DURATION = min(MAX_DURATION, MAX_DURATION_LIMIT)
    
    MAX_STEPS = int(MAX_DURATION / TIME_RESOLUTION)  # 시간에 맞게 steps 계산
    print(f"🔍 Max duration: {MAX_DURATION:.2f}s, Max steps: {MAX_STEPS}")

    roll = np.zeros((MAX_STEPS, 128), dtype=np.float32)
    print(f"📦 Roll shape: {roll.shape}, Estimated size: {roll.nbytes / 1024 / 1024:.2f} MB")
    
    for note in notes:
        time_sec = note["time"]
        pitch = note["pitch"]
        velocity = note["velocity"] / 127.0
        
        step = int(time_sec / TIME_RESOLUTION)
        if 0 <= step < MAX_STEPS and 0 <= pitch < 128:
            roll[step][pitch] = velocity
    
    # 배열 크기 및 메모리 사용량 출력
    print(f"📊 Roll shape: {roll.shape}")
    print(f"🧠 Memory size of the roll array: {sys.getsizeof(roll) / (1024 * 1024):.2f} MB")
    
    return roll

def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    total_files = len(files)
    
    for idx, fname in enumerate(files):
        save_path = os.path.join(OUTPUT_DIR, fname.replace(".json", ".npz"))
        
        # 이미 처리된 파일은 건너뛰기
        if os.path.exists(save_path):
            print(f"✅ {fname} has already been processed. Skipping.")
            continue
        
        full_path = os.path.join(INPUT_DIR, fname)
        print(f"🎼 Processing {idx + 1}/{total_files}: {fname}")
        
        try:
            with open(full_path, "r") as f:
                notes = json.load(f)
            
            # 최대 시간과 피아노 롤 계산을 한번에
            roll = notes_to_piano_roll(notes)
            
            # 압축하여 .npz 파일로 저장
            np.savez_compressed(save_path, roll=roll)
            print(f"✅ Processed and saved {fname}")
            
            # 피아노 롤 처리 후 메모리 비우기
            del roll  # 피아노 롤 객체 삭제
            gc.collect()  # 가비지 컬렉터 호출하여 메모리 정리

        except Exception as e:
            print(f"❌ Failed on {fname}: {e}")
            # 실패한 파일은 계속 진행하기 위해 skip

if __name__ == "__main__":
    main()
