import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
import gc  # Garbage Collection 라이브러리

# 하이퍼파라미터
TIME_RESOLUTION = 0.05  # 50ms 단위 (20 steps/sec)
LATENT_DIM = 32  # 잠재 공간 차원
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
SEQUENCE_LENGTH = 200  # 예시로 200 타임스텝 설정

# GPU 설정: TensorFlow가 GPU를 자동으로 인식하므로 설정 필요
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')  # 첫 번째 GPU 선택
    tf.config.experimental.set_memory_growth(physical_devices[0], True)  # 메모리 증가 설정

# 피아노 롤 데이터 로딩 (슬라이딩 윈도우 적용)
def load_data(input_dir="data/processed/Classical", seq_length=200, step=100):
    """피아노 롤 데이터를 불러와서 슬라이딩 윈도우로 시퀀스 생성"""
    files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]
    sequences = []
    
    for file in files:
        file_path = os.path.join(input_dir, file)
        try:
            with np.load(file_path) as npz_file:
                roll = npz_file['roll']
                
                # 시퀀스 길이보다 짧은 롤은 무시
                if roll.shape[0] < seq_length:
                    continue
                    
                # 슬라이딩 윈도우로 시퀀스 생성
                for i in range(0, roll.shape[0] - seq_length + 1, step):
                    seq = roll[i:i + seq_length]
                    sequences.append(seq)
        except Exception as e:
            print(f"Error loading or processing file {file}: {e}")

    return np.array(sequences)

# VAE 인코더
def build_encoder(input_shape, latent_dim):
    """VAE 인코더 모델"""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)

    # 잠재 공간에 대한 평균(mean)과 표준편차(standard deviation)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    # z_mean과 z_log_var를 이용해 잠재 공간 샘플링
    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder

# 샘플링 함수 (Reparameterization trick)
def sampling(args):
    """잠재 공간에서 샘플링"""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    return z

# VAE 디코더
def build_decoder(latent_dim, output_shape):
    """VAE 디코더 모델"""
    latent_inputs = layers.Input(shape=(latent_dim,))
    
    x = layers.Dense(64)(latent_inputs)
    x = layers.RepeatVector(SEQUENCE_LENGTH)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    
    outputs = layers.TimeDistributed(layers.Dense(output_shape[1], activation='sigmoid'))(x)
    
    decoder = models.Model(latent_inputs, outputs, name='decoder')
    return decoder

# 전체 VAE 모델
def build_vae(input_shape, latent_dim):
    """VAE 모델"""
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape)

    # VAE 인코더와 디코더 연결
    inputs = layers.Input(shape=input_shape)
    z_mean, z_log_var, z = encoder(inputs)
    reconstructed = decoder(z)

    # 손실 함수 정의 (재구성 손실 + KL 발산)
    vae = models.Model(inputs, reconstructed, name='vae')

    # VAE 손실 함수 정의
    reconstruction_loss = tf.keras.ops.mean(tf.keras.ops.square(inputs - reconstructed))
    kl_loss = -0.5 * tf.keras.ops.mean(tf.keras.ops.sum(1 + z_log_var - tf.keras.ops.square(z_mean) - tf.keras.ops.exp(z_log_var), axis=-1))
    vae_loss = reconstruction_loss + kl_loss

    vae.add_loss(vae_loss)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    return vae, encoder, decoder

# 학습 데이터 로드
data = load_data(seq_length=SEQUENCE_LENGTH)  # 피아노 롤 데이터 로드
input_shape = (SEQUENCE_LENGTH, 128)  # 시퀀스 길이, 피치 수

# VAE 모델 생성
vae, encoder, decoder = build_vae(input_shape, LATENT_DIM)

# 학습 도중 출력할 샘플을 생성하는 콜백 함수
def generate_sample(epoch, logs):
    """학습 중 중간 결과 생성"""
    if epoch % 10 == 0:  # 10번째 에포크마다 출력
        random_latent_vector = np.random.normal(size=(1, LATENT_DIM))  # 잠재 벡터 샘플링
        generated_roll = decoder.predict(random_latent_vector)  # 디코더를 통해 음악 생성
        
        # 생성된 피아노 롤을 MIDI 파일로 변환하여 저장
        #save_to_midi(generated_roll[0], f'generated_sample_epoch_{epoch}.mid')  # 출력 파일 저장
        print(f"Generated sample at epoch {epoch} saved as MIDI")

# 콜백 정의
generate_sample_callback = LambdaCallback(on_epoch_end=generate_sample)

# 모델 체크포인트 콜백 (모델 저장)
checkpoint_callback = ModelCheckpoint('vae_model_checkpoint.h5', save_best_only=True, monitor='loss', mode='min')

if __name__ == '__main__':
    # 모델 학습
    vae.fit(data, data, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint_callback, generate_sample_callback])

    # 모델 저장
    vae.save('vae_model.h5')
    encoder.save('encoder_model.h5')
    decoder.save('decoder_model.h5')

# 모델을 이용해 새로운 피아노 롤 생성하기
def generate_music():
    """생성된 음악 샘플"""
    # 잠재 공간에서 샘플링
    random_latent_vector = np.random.normal(size=(1, LATENT_DIM))
    generated_roll = decoder.predict(random_latent_vector)

    # 생성된 피아노 롤을 MIDI로 변환 (여기서는 MIDI 변환 함수 필요)
    # save_to_midi(generated_roll)
    return generated_roll

# 생성된 음악 샘플 출력
generated_music = generate_music()
print(generated_music)
