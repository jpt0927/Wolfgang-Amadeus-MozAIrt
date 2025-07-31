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

EPOCHS = 1000

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
# 1. 설정 및 머니코드 라이브러리
# ======================================

# 🎵 수십 개의 '머니코드' 진행을 로마 숫자로 정의
#   - 팝, 록, 발라드 등에서 매우 자주 쓰이는 진행들의 목록입니다.
#   - 얼마든지 원하는 진행을 추가할 수 있습니다.
MONEY_PROGRESSIONS_RN = [
    # 클래식 캐논 진행 (매우 유명)
    ['I', 'V', 'vi', 'iii', 'IV', 'I', 'IV', 'V'],
    # 팝 펑크 진행
    ['I', 'V', 'vi', 'IV'],
    # 50년대 두왑 진행
    ['I', 'vi', 'IV', 'V'],
    # 블루스/록의 기본 12마디 블루스 (기본)
    ['I', 'I', 'I', 'I', 'IV', 'IV', 'I', 'I', 'V', 'IV', 'I', 'I'],
    # 마이너 키의 인기 진행
    ['vi', 'IV', 'I', 'V'],
    ['i', 'VI', 'III', 'VII'],
    # 재즈의 기본
    ['ii', 'V', 'I'],
    # 그 외 다양한 인기 진행
    ['I', 'IV', 'V', 'I'],
    ['I', 'IV', 'vi', 'V'],
    ['I', 'V', 'IV', 'V'],
]

# ⚙️ 후처리 관련 하이퍼파라미터
WINDOW_DURATION_SECONDS = 0.8  # 코드를 감지할 시간 창의 길이 (초)
TIME_RESOLUTION = 0.05         # 피아노 롤의 시간 해상도
NOTE_THRESHOLD = 0.5           # 노트로 인식할 최소 강도
REDUCTION_FACTOR = 0.6         # 머니코드가 아닌 부분의 세기를 줄일 비율 (0.6 = 60%로)

# ======================================
# 2. 핵심 유틸리티 함수
# ======================================
# --- 키(Key) 분석 관련 ---
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def detect_key(piano_roll):
    """피아노 롤의 키(Key)를 Krumhansl-Schmuckler 알고리즘으로 추정합니다."""
    chroma = np.sum(piano_roll, axis=0) # 각 Pitch별 총 강도
    chromagram = np.zeros(12)
    for pitch, intensity in enumerate(chroma):
        chromagram[pitch % 12] += intensity
    
    if np.sum(chromagram) == 0: return "C", "Major" # 음이 없는 경우 기본값

    correlations = []
    for i in range(12):
        # Major keys
        corr_maj = np.corrcoef(chromagram, np.roll(MAJOR_PROFILE, i))[0, 1]
        correlations.append((PITCH_NAMES[i], 'Major', corr_maj))
        # Minor keys
        corr_min = np.corrcoef(chromagram, np.roll(MINOR_PROFILE, i))[0, 1]
        correlations.append((PITCH_NAMES[i], 'minor', corr_min))

    best_match = max(correlations, key=lambda x: x[2])
    return best_match[0], best_match[1]

# --- 코드 분석 관련 ---
CHORD_TEMPLATES = {
    'maj': {0, 4, 7}, 'min': {0, 3, 7}, 'dim': {0, 3, 6},
    'aug': {0, 4, 8}, 'maj7': {0, 4, 7, 11}, 'min7': {0, 3, 7, 10},
    'dom7': {0, 4, 7, 10}, 'dim7': {0, 3, 6, 9}
}
RN_MAJOR = {'I':'maj', 'ii':'min', 'iii':'min', 'IV':'maj', 'V':'maj', 'vi':'min', 'vii°':'dim'}
RN_MINOR = {'i':'min', 'ii°':'dim', 'III':'maj', 'iv':'min', 'v':'min', 'VI':'maj', 'VII':'maj'}
KEY_OFFSETS = {'I':0, 'ii':2, 'iii':4, 'IV':5, 'V':7, 'vi':9, 'vii°':11, 'i':0, 'ii°':2, 'III':3, 'iv':5, 'v':7, 'VI':8, 'VII':10}


def detect_chord_in_window(window, threshold):
    """피아노 롤 윈도우에서 가장 유사한 코드를 찾습니다."""
    active_notes = window > threshold
    if not np.any(active_notes): return None

    active_pitches = {p % 12 for p, t in np.argwhere(active_notes)}
    if len(active_pitches) < 2: return None

    # 모든 루트(C, C#, ...)와 모든 코드 타입(maj, min, ...)에 대해 비교
    best_match = None
    max_similarity = 0
    for root in range(12):
        for quality, template in CHORD_TEMPLATES.items():
            template_pitches = {(p + root) % 12 for p in template}
            intersection = len(active_pitches.intersection(template_pitches))
            union = len(active_pitches.union(template_pitches))
            if union == 0: continue
            
            similarity = intersection / union
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = f"{PITCH_NAMES[root]}{quality}"
    
    if max_similarity > 0.35: # 특정 유사도 이상일 때만 인정
        return best_match
    return None

def to_roman_numeral(chord_name, key, quality):
    """코드 이름을 키(Key)에 맞춰 로마 숫자로 변환합니다."""
    if not chord_name: return None
    key_root = PITCH_NAMES.index(key)
    chord_root_str = chord_name.split('m')[0].split('d')[0].split('a')[0] # C, C#, D...
    chord_root = PITCH_NAMES.index(chord_root_str)
    chord_quality = 'maj' if 'maj' in chord_name else 'min' if 'min' in chord_name else 'dim' if 'dim' in chord_name else 'maj'

    interval = (chord_root - key_root + 12) % 12
    
    rn_map = RN_MAJOR if quality == 'Major' else RN_MINOR
    for rn, offset in KEY_OFFSETS.items():
        if rn not in rn_map: continue
        if interval == offset and rn_map[rn] == chord_quality:
            return rn
    return None # 해당 키의 다이어토닉 코드가 아님


# ======================================
# 3. 메인 후처리 함수
# ======================================

def post_process_with_many_chords(piano_roll):
    """
    수십 개의 머니코드 진행 목록을 기반으로 피아노 롤을 후처리합니다.
    1. 키 분석 -> 2. 코드 감지 -> 3. 로마숫자 변환 -> 4. 머니코드 매칭 -> 5. 강도 조절
    """
    print("\n🎹 개선된 후처리 시작 (키 분석 기반)...")
    
    # 1. 키 분석
    key, key_quality = detect_key(piano_roll)
    print(f"-> 감지된 키: {key} {key_quality}")
    
    # 2. 코드 감지 및 3. 로마숫자 변환
    window_size_steps = int(WINDOW_DURATION_SECONDS / TIME_RESOLUTION)
    num_windows = len(piano_roll) // window_size_steps
    
    detected_rn_sequence = []
    for i in range(num_windows):
        start = i * window_size_steps
        end = start + window_size_steps
        window = piano_roll[start:end, :]
        chord_name = detect_chord_in_window(window, NOTE_THRESHOLD)
        roman_numeral = to_roman_numeral(chord_name, key, key_quality)
        detected_rn_sequence.append(roman_numeral)

    print(f"-> 감지된 로마 숫자 시퀀스: {[rn for rn in detected_rn_sequence if rn]}")
    
    # 4. 머니코드 매칭
    is_money_chord_window = [False] * num_windows
    found_count = 0
    
    # 모든 머니코드 진행 목록을 순회하며 매칭
    for progression in MONEY_PROGRESSIONS_RN:
        prog_len = len(progression)
        for i in range(num_windows - prog_len + 1):
            sequence_to_check = detected_rn_sequence[i : i + prog_len]
            if sequence_to_check == progression:
                found_count += 1
                # 해당 윈도우들을 '머니코드' 구간으로 표시
                for j in range(prog_len):
                    is_money_chord_window[i + j] = True

    if found_count > 0:
        print(f"-> 총 {found_count}개의 머니코드 진행을 발견했습니다!")
    else:
        print("-> 머니코드 진행을 찾지 못했습니다. 원본을 그대로 반환합니다.")
        return piano_roll
        
    # 5. 강도 조절
    weight_mask = np.full(piano_roll.shape, REDUCTION_FACTOR)
    for i, is_money in enumerate(is_money_chord_window):
        if is_money:
            start_step = i * window_size_steps
            end_step = start_step + window_size_steps
            weight_mask[start_step:end_step, :] = 1.0

    processed_roll = piano_roll * weight_mask
    print("후처리 완료!")
    return processed_roll




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

            processed_roll = post_process_with_many_chords(trimmed_roll)

           

            print(f"\n--- 생성된 음악 정보 ---")

            print(f"원본 길이: {len(new_music_roll)} 스텝 (~{len(new_music_roll)*0.05:.1f}초)")

            print(f"무음 제거 후 길이: {len(processed_roll)} 스텝 (~{len(processed_roll)*0.05:.1f}초)")

            print(f"활성화된 노트 수 (0.5 이상): {np.sum(processed_roll > 0.5)}개")



            # 3. 피아노 롤 시각화 (PNG 저장)

            print("\n피아노 롤 시각화 중...")

            plt.figure(figsize=(12, 8))

            plt.imshow(processed_roll.T > 0.5, aspect='auto', origin='lower', cmap='gray_r')

            plt.title('Generated Piano Roll')

            plt.xlabel('Time Step')

            plt.ylabel('MIDI Pitch')

            plt.savefig(OUTPUT_IMG_PATH)

            print(f"피아노 롤 이미지를 '{OUTPUT_IMG_PATH}'에 저장했습니다.")

           

            # 4. MIDI 파일로 저장

            save_roll_to_midi(trimmed_roll, OUTPUT_MIDI_PATH)

           

        except Exception as e:

            print(f"음악 생성 중 오류가 발생했습니다: {e}")