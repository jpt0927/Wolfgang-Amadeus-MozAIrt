# model/preprocess.py

import numpy as np
import pretty_midi
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
import model # model.py의 변수들을 가져오기 위함

def preprocess_and_save():
    """
    원본 MIDI 파일들을 읽어 패딩된 피아노 롤로 변환 후,
    하나의 압축된 numpy 파일(.npz)로 저장합니다.
    """
    print(f"'{model.DATA_DIR}' 경로에서 MIDI 파일 전처리를 시작합니다...")
    
    # model.py의 load_data 함수 로직을 그대로 사용
    input_dir = Path(model.DATA_DIR)
    files = list(input_dir.glob("*.mid*"))
    if not files:
        raise RuntimeError(f"MIDI 파일이 없습니다: {input_dir}")
        
    all_rolls = []
    fs = 1 / model.TIME_RESOLUTION
    
    for fp in files:
        try:
            midi_data = pretty_midi.PrettyMIDI(str(fp))
            roll = midi_data.get_piano_roll(fs=fs)
            normalized_roll = (roll.T / 127.0).astype(np.float32)
            all_rolls.append(normalized_roll)
        except Exception as e: 
            print(f"[WARN] {fp.name} 처리 중 오류: {e}")

    padded_rolls = pad_sequences(
        all_rolls, maxlen=model.MAX_SEQUENCE_LENGTH, padding='post', truncating='post', dtype='float32'
    )
    
    final_data = np.expand_dims(padded_rolls, -1)
    
    # 전처리된 데이터를 저장할 경로
    save_path = model.DATA_DIR.parent / f"{model.DATASET_TAG}_preprocessed.npz"
    
    np.savez_compressed(save_path, data=final_data)
    print(f"\n✅ 전처리 완료! 데이터가 '{save_path}'에 저장되었습니다.")
    print(f"최종 데이터 형태: {final_data.shape}")

if __name__ == "__main__":
    preprocess_and_save()