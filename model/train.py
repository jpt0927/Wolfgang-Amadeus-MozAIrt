# train.py
import model
import numpy as np

if __name__ == "__main__":
    model.set_gpu()

    print(f"모델 학습 시작 ({model.DATASET_TAG})...")
    
    # 1. 전처리 스크립트 실행을 권장하는 부분은 그대로 둡니다.
    preprocessed_path = model.DATA_DIR.parent / f"{model.DATASET_TAG}_preprocessed.npz"
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"전처리된 데이터 파일이 없습니다: {preprocessed_path}\n"
            "먼저 'python3 model/preprocess.py'를 실행하여 데이터 파일을 생성해주세요."
        )
    print(f"전처리된 데이터 로딩 중: {preprocessed_path}")
    with np.load(preprocessed_path) as npz:
        data = npz['data']
    
    if np.isnan(data).any():
        raise ValueError("데이터에 NaN 값이 포함되어 있습니다.")
        
    input_shape = data.shape[1:]
    
    # 2. 모델 생성
    print("모델 생성 중...")
    encoder = model.build_encoder(input_shape, model.LATENT_DIM)
    decoder = model.build_decoder(model.LATENT_DIM, encoder.shape_before_flatten)
    # ✅ build_and_compile_vae 함수로 한 번에 처리
    vae = model.build_and_compile_vae(encoder, decoder)
    
    print("모델 구조:")
    vae.summary()

    vae.build(input_shape=(None, *input_shape))
    
    print("\n모델 학습 시작...")
    # ✅ fit 함수에 x, y를 모두 전달
    history = vae.fit(
        x=data, 
        y=data,
        epochs=model.EPOCHS,
        batch_size=model.BATCH_SIZE,
        callbacks=[model.checkpoint_callback],
        verbose=1,
    )

    print("\n모델 학습 완료")

    # 4. 저장
    vae.encoder.save(str(model.ENC_SAVE_PATH))
    vae.decoder.save(str(model.DEC_SAVE_PATH))
    vae.save_weights(str(model.VAE_SAVE_PATH).replace('.keras', '.weights.h5'))

    print("\n[SAVED]")
    print(f" VAE     : {str(model.VAE_SAVE_PATH).replace('.keras', '.weights.h5')}")
    print(f" Encoder : {model.ENC_SAVE_PATH}")
    print(f" Decoder : {model.DEC_SAVE_PATH}")