import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. 기본 설정 (학습 시와 동일하게) ---
LATENT_DIM = 32
SEQUENCE_LENGTH = 200
OUTPUT_PITCH_DIM = 128

# --- 2. VAE 모델 전체 구조 정의 (학습 코드에서 가져옴) ---
def sampling(args):
    """Reparameterization trick."""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape, latent_dim):
    """VAE Encoder Model"""
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    return models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

def build_decoder(latent_dim, seq_length, output_dim):
    """VAE Decoder Model"""
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64)(latent_inputs)
    x = layers.RepeatVector(seq_length)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(output_dim, activation='sigmoid'))(x)
    return models.Model(latent_inputs, outputs, name='decoder')

def build_vae(input_shape, latent_dim):
    """Builds the full VAE model."""
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape[0], input_shape[1])
    
    inputs = layers.Input(shape=input_shape)
    _, _, z = encoder(inputs)
    reconstructed = decoder(z)
    
    vae = models.Model(inputs, reconstructed, name='vae')
    return vae, encoder, decoder

# --- 3. 음악 생성 함수 (이전과 동일) ---
def generate_music(decoder_model, latent_dim):
    """학습된 디코더를 사용해 새로운 음악 생성"""
    random_latent_vector = np.random.normal(size=(1, latent_dim))
    generated_roll = decoder_model.predict(random_latent_vector)
    return generated_roll[0]

# --- 4. MIDI 저장을 위한 유틸리티 함수 ---
def save_roll_to_midi(piano_roll, filename, threshold=0.5, time_resolution=0.05):
    # 이 함수를 사용하려면 pretty_midi 라이브러리가 필요합니다.
    import pretty_midi
    
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0) # 0: Acoustic Grand Piano

    # 활성화 확률이 threshold 이상인 노트를 실제 노트로 변환
    binary_roll = piano_roll > threshold
    
    for pitch in range(binary_roll.shape[1]):
        start_time = None
        for time_step in range(binary_roll.shape[0]):
            time = time_step * time_resolution
            if binary_roll[time_step, pitch] and start_time is None:
                start_time = time
            elif not binary_roll[time_step, pitch] and start_time is not None:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start_time,
                    end=time
                )
                instrument.notes.append(note)
                start_time = None

    pm.instruments.append(instrument)
    pm.write(filename)
    print(f"\n🎵 생성된 음악을 '{filename}' 파일로 저장했습니다.")

# --- 5. 메인 실행 부분 ---
if __name__ == '__main__':
    MODEL_PATH = '../vae_model_checkpoint.h5'

    if not os.path.exists(MODEL_PATH):
        print(f"오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다.")
    else:
        try:
            input_shape = (SEQUENCE_LENGTH, OUTPUT_PITCH_DIM)
            vae_model, _, decoder = build_vae(input_shape, LATENT_DIM)
            vae_model.load_weights(MODEL_PATH)

            print("\n✅ 모델 가중치 로드 및 디코더 추출 성공!")

            new_music_roll = generate_music(decoder, LATENT_DIM)
            
            print("\n🎹 새로 생성된 피아노 롤 데이터:")
            print(f"Shape: {new_music_roll.shape}")

            print("\n--- 모델 출력 데이터 진단 시작 ---")
            print(f"Shape: {new_music_roll.shape}")
            print(f"Max value: {np.max(new_music_roll)}")
            print(f"Min value: {np.min(new_music_roll)}")
            print(f"Mean value: {np.mean(new_music_roll)}")
            print(f"Number of 'notes' above threshold 0.5: {np.sum(new_music_roll > 0.5)}")
            print("--- 모델 출력 데이터 진단 종료 ---\n")
            
            # --- MIDI 파일 저장 함수 호출 ---
            save_roll_to_midi(new_music_roll, "generated_music.mid")
            
        except Exception as e:
            print(f"모델 로딩 또는 음악 생성 중 오류가 발생했습니다: {e}")