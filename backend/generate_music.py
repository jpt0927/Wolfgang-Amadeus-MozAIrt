# 최종 수정 버전: 이 코드는 단일 트랙 MIDI 파일을 생성합니다.
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pretty_midi

# --- 기본 설정 ---
LATENT_DIM = 32
SEQUENCE_LENGTH = 200
OUTPUT_PITCH_DIM = 128

# --- VAE 모델 구조 정의 ---
def build_vae(input_shape, latent_dim):
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(encoder_inputs)
    x = layers.LSTM(64)(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64)(latent_inputs)
    x = layers.RepeatVector(input_shape[0])(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    decoder_outputs = layers.TimeDistributed(layers.Dense(input_shape[1], activation='sigmoid'))(x)
    decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')

    # Full VAE
    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae = models.Model(encoder_inputs, vae_outputs, name='vae')
    
    return vae, encoder, decoder

# --- 음악 생성 함수 ---
def generate_music(decoder_model, latent_dim):
    random_latent_vector = np.random.normal(size=(1, latent_dim))
    generated_roll = decoder_model.predict(random_latent_vector)
    return generated_roll[0]

# --- 최종 MIDI 저장 함수 ---
def save_roll_to_midi(piano_roll, filename, threshold=0.5, time_resolution=0.05):
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
                    note = pretty_midi.Note(
                        velocity=100, pitch=pitch, start=start_time, end=end_time
                    )
                    instrument.notes.append(note)

    # 루프가 끝난 후에도 'on' 상태인 롱 노트 처리
    if notes_on:
        end_time = len(piano_roll) * time_resolution
        for pitch, start_time in notes_on.items():
            if end_time > start_time:
                note = pretty_midi.Note(
                    velocity=100, pitch=pitch, start=start_time, end=end_time
                )
                instrument.notes.append(note)

    midi_data.instruments.append(instrument)
    midi_data.write(filename)
    print(f"\n🎵 생성된 음악을 '{filename}' 파일로 저장했습니다.")


# --- 메인 실행 부분 ---
if __name__ == '__main__':
    MODEL_WEIGHTS_PATH = '../vae_Electronic_Dance_SynthPop_E20.h5'

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"오류: 모델 가중치 파일 '{MODEL_WEIGHTS_PATH}'을 찾을 수 없습니다.")
    else:
        try:
            input_shape = (SEQUENCE_LENGTH, OUTPUT_PITCH_DIM)
            vae, _, decoder = build_vae(input_shape, LATENT_DIM)
            vae.load_weights(MODEL_WEIGHTS_PATH)

            print("\n✅ 모델 가중치 로드 성공!")

            new_music_roll = generate_music(decoder, LATENT_DIM)
            
            save_roll_to_midi(new_music_roll, "generated_music.mid")
            
        except Exception as e:
            print(f"음악 생성 중 오류가 발생했습니다: {e}")