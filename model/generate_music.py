import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import pretty_midi
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================
# 설정 (본인의 환경에 맞게 수정)
# ======================================
# 학습 시 사용했던 하이퍼파라미터
LATENT_DIM = 128
EPOCHS = 200
DATASET_TAG = "Rock_Pop"

# 프로젝트 최상위 경로
try:
    ROOT_DIR = Path(__file__).resolve().parents[1]
except NameError:
    ROOT_DIR = Path.cwd()

# 1. 사용할 디코더 모델 파일 경로
DECODER_PATH = ROOT_DIR / "outputs" / f"{DATASET_TAG}_CNN_Padding_Midi" / f"decoder_{DATASET_TAG}_E{EPOCHS}.keras"

# 2. 생성된 파일이 저장될 경로
OUTPUT_MIDI_PATH = ROOT_DIR / "outputs" / "generated_music.mid"
OUTPUT_IMG_PATH = ROOT_DIR / "outputs" / "piano_roll_visualization.png"

# ======================================
# 핵심 함수
# ======================================

def trim_silence(piano_roll, threshold=0.1):
    """피아노 롤의 뒷부분에 있는 무음 구간을 제거합니다."""
    # (시간, 음높이) 축에서 음이 하나라도 있는지 확인
    has_note = np.sum(piano_roll, axis=1) > threshold
    if np.any(has_note):
        # 마지막 음이 있는 시간 스텝 찾기
        last_note_step = np.where(has_note)[0][-1]
        return piano_roll[:last_note_step + 1, :]
    else:
        # 아무 음도 없는 경우 그대로 반환
        return piano_roll

def save_roll_to_midi(piano_roll, filename, time_resolution=0.05, threshold=0.5):
    """피아노 롤을 MIDI 파일로 저장합니다."""
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    notes_on = {}
    for time_step, frame in enumerate(piano_roll):
        current_time = time_step * time_resolution
        for pitch in range(128):
            is_on = frame[pitch] >= threshold
            if is_on and pitch not in notes_on:
                notes_on[pitch] = current_time
            elif not is_on and pitch in notes_on:
                start_time = notes_on.pop(pitch)
                end_time = current_time
                if end_time > start_time:
                    instrument.notes.append(
                        pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
                    )
    if notes_on:
        end_time = len(piano_roll) * time_resolution
        for pitch, start_time in notes_on.items():
            if end_time > start_time:
                instrument.notes.append(
                    pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
                )
    midi_data.instruments.append(instrument)
    midi_data.write(str(filename))
    print(f"\n🎵 생성된 음악을 '{filename}' 파일로 저장했습니다.")

# ======================================
# 메인 실행 로직
# ======================================
if __name__ == '__main__':
    if not DECODER_PATH.exists():
        print(f"오류: 디코더 모델 파일을 찾을 수 없습니다: {DECODER_PATH}")
    else:
        try:
            print("디코더 모델 로딩 중...")
            decoder = models.load_model(str(DECODER_PATH), compile=False)
            decoder.summary()

            print("\n음악 생성 중...")
            random_latent_vector = np.random.normal(size=(1, LATENT_DIM))
            
            # 1. 디코더로 피아노 롤 생성
            generated_roll = decoder.predict(random_latent_vector)
            new_music_roll = np.squeeze(generated_roll, axis=(0, -1))

            # 2. 뒷부분의 불필요한 سکوت 구간 제거
            trimmed_roll = trim_silence(new_music_roll)
            
            print(f"\n--- 생성된 음악 정보 ---")
            print(f"원본 길이: {len(new_music_roll)} 스텝 (~{len(new_music_roll)*0.05:.1f}초)")
            print(f"무음 제거 후 길이: {len(trimmed_roll)} 스텝 (~{len(trimmed_roll)*0.05:.1f}초)")
            print(f"활성화된 노트 수 (0.5 이상): {np.sum(trimmed_roll > 0.5)}개")

            # 3. 피아노 롤 시각화 (PNG 저장)
            print("\n피아노 롤 시각화 중...")
            plt.figure(figsize=(12, 8))
            plt.imshow(trimmed_roll.T > 0.5, aspect='auto', origin='lower', cmap='gray_r')
            plt.title('Generated Piano Roll')
            plt.xlabel('Time Step')
            plt.ylabel('MIDI Pitch')
            plt.savefig(OUTPUT_IMG_PATH)
            print(f"피아노 롤 이미지를 '{OUTPUT_IMG_PATH}'에 저장했습니다.")
            
            # 4. MIDI 파일로 저장
            save_roll_to_midi(trimmed_roll, OUTPUT_MIDI_PATH)
            
        except Exception as e:
            print(f"음악 생성 중 오류가 발생했습니다: {e}")