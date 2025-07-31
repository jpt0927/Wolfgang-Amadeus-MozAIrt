#그냥 midi 파일보다 성능이 안좋게 전처리해도 모델이 견디도록 하는 스파르타식 학습

import os
import json
from mido import MidiFile

RAW_DIR = 'data/raw/Classical'
SAVE_DIR = 'data/extracted/Classical'
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_notes(midi_path):
    try:
        midi = MidiFile(midi_path)
        abs_time = 0
        notes = []

        # 모델이 학습하는데 필요한 pitch, velocity, time만 추출
        for track in midi.tracks:
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append({
                        'pitch': msg.note,
                        'velocity': msg.velocity,
                        'time': abs_time
                    })

        return notes

    except Exception as e:
        print(f"❌ Error in {midi_path}: {e}")
        return None

def main():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.mid')]
    for f in files:
        full_path = os.path.join(RAW_DIR, f)
        print(f"🎵 Extracting: {f}")
        notes = extract_notes(full_path)

        if notes:
            save_path = os.path.join(SAVE_DIR, f.replace('.mid', '.json'))
            with open(save_path, 'w') as out_file:
                json.dump(notes, out_file)
            print(f"✅ Saved to: {save_path}")
        else:
            print(f"⚠️  Skipped {f} due to error")

if __name__ == '__main__':
    main()
