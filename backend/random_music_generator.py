import random
import numpy as np
import pretty_midi

# --- 설정 ---
TIME_RESOLUTION = 0.05  # 각 타임스텝의 시간 (초)
SEQUENCE_LENGTH = 600   # 600 * 0.05 = 30초
OUTPUT_FILENAME = "generated_music.mid"

# --- 음악 데이터 정의 ---
CHORDS = {
    'Cmaj7': [60, 64, 67, 71], 'Dm7': [62, 65, 69, 72], 'Em7': [64, 67, 71, 74],
    'Fmaj7': [65, 69, 72, 76], 'G7': [67, 71, 74, 78],  'Am7': [69, 72, 76, 79]
}
CHORD_PALETTE = list(CHORDS.keys())

RHYTHM_PATTERNS = [
    [(4, 0), (4, 0), (4, 0), (4, 0)],
    [(6, 0), (2, 0), (6, 0), (2, 0)],
    [(2, 2), (2, 2), (2, 2), (2, 2)],
    [(3, 1), (3, 1), (3, 1), (3, 1)]
]

MELODY_PATTERNS = {
    'up': lambda c: c,
    'down': lambda c: c[::-1],
    'up_down': lambda c: [c[0], c[2], c[1], c[3]],
}
MELODY_PATTERN_NAMES = list(MELODY_PATTERNS.keys())

# --- 핵심 함수 ---
def generate_piano_roll():
    """
    랜덤 조합으로 음악을 생성하되, 마지막은 V-I 진행으로 마무리합니다.
    """
    roll = np.zeros((SEQUENCE_LENGTH, 128), dtype=np.float32)
    current_step = 0
    progression_log = []
    
    # 마지막 마무리를 위한 공간(80스텝, 4초)을 남겨둠
    ending_length = 80
    
    # 1. 마지막 부분을 제외하고 랜덤 코드 진행 생성
    while current_step < SEQUENCE_LENGTH - ending_length:
        chord_name = random.choice(CHORD_PALETTE)
        measure_length = random.randint(40, 80)

        if current_step + measure_length > SEQUENCE_LENGTH - ending_length:
            measure_length = SEQUENCE_LENGTH - ending_length - current_step
        if measure_length <= 0:
            break
            
        progression_log.append(chord_name)
        
        # (이전과 동일한 베이스/아르페지오 생성 로직)
        chord_notes = CHORDS[chord_name]
        bass_note = chord_notes[0] - 24
        roll[current_step : current_step + measure_length, bass_note] = 0.8
        rhythm_pattern = random.choice(RHYTHM_PATTERNS)
        melody_pattern = MELODY_PATTERNS[random.choice(MELODY_PATTERN_NAMES)](chord_notes)
        time_in_measure = 0
        pattern_index = 0
        while time_in_measure < measure_length:
            note_len, rest_len = rhythm_pattern[pattern_index % len(rhythm_pattern)]
            if time_in_measure + note_len > measure_length: break
            note_pitch = melody_pattern[pattern_index % len(melody_pattern)] + 12
            note_start_step = current_step + time_in_measure
            roll[note_start_step : note_start_step + note_len, note_pitch] = 1.0
            time_in_measure += note_len + rest_len
            pattern_index += 1
        current_step += measure_length

    # ############ 여기가 수정된 부분 ############
    # 2. 마지막 4초는 G7 -> Cmaj7 코드로 마무리
    print("마무리 코드 진행: G7 -> Cmaj7")
    
    # G7 코드 (긴장)
    g7_chord = CHORDS['G7']
    g7_length = ending_length // 2
    # 베이스
    roll[current_step : current_step + g7_length, g7_chord[0] - 24] = 0.8
    # 아르페지오
    for j in range(4):
        roll[current_step + j*10 : current_step + j*10 + 8, g7_chord[j] + 12] = 1.0
    current_step += g7_length
    
    # Cmaj7 코드 (해소)
    cmaj7_chord = CHORDS['Cmaj7']
    # 베이스를 길게 연주하며 마무리
    roll[current_step : SEQUENCE_LENGTH, cmaj7_chord[0] - 24] = 0.8
    # 마지막 화음을 길게 연주하며 마무리
    for note_pitch in cmaj7_chord:
        roll[current_step : SEQUENCE_LENGTH, note_pitch] = 1.0
    # ############ 여기까지 ############

    print(f"생성된 코드 진행: {' -> '.join(progression_log)} -> G7 -> Cmaj7")
    return roll

def save_roll_to_midi(piano_roll, filename, threshold=0.5):
    """
    피아노 롤을 MIDI 파일로 저장하는 함수
    """
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    notes_on = {}
    for time_step, frame in enumerate(piano_roll):
        current_time = time_step * TIME_RESOLUTION
        for pitch in range(128):
            is_on = frame[pitch] >= threshold
            if is_on and pitch not in notes_on:
                notes_on[pitch] = current_time
            elif not is_on and pitch in notes_on:
                start_time = notes_on.pop(pitch)
                end_time = current_time
                if end_time > start_time:
                    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time))
    if notes_on:
        end_time = len(piano_roll) * TIME_RESOLUTION
        for pitch, start_time in notes_on.items():
            if end_time > start_time:
                instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time))
    midi_data.instruments.append(instrument)
    midi_data.write(filename)
    print(f"\n🎵 생성된 음악을 '{filename}' 파일로 저장했습니다.")

# --- 메인 실행 ---
if __name__ == "__main__":
    print("완전 랜덤 조합 음악 생성을 시작합니다...")
    piano_roll_data = generate_piano_roll()
    save_roll_to_midi(piano_roll_data, OUTPUT_FILENAME)
    print("\n작업 완료!")