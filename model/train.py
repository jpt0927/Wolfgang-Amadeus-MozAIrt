# train.py
import model

if __name__ == "__main__":
    model.set_gpu()

    print("모델 학습 시작 (패딩/마스킹 방식)...")

    print("데이터 로딩 및 패딩 중...")
    data = model.load_data()
    # input_shape은 데이터 로딩 후 결정
    input_shape = data.shape[1:]
    
    print(f"데이터 로드 완료. 최종 데이터 형태: {data.shape}")

    print("모델 생성 중...")
    encoder = model.build_encoder(input_shape, model.LATENT_DIM)
    decoder = model.build_decoder(model.LATENT_DIM, encoder.shape_before_flatten)
    vae = model.build_vae(encoder, decoder)
    
    print("모델 구조:")
    vae.summary()

    # 학습 진행
    history = vae.fit(
        data, data,
        epochs=model.EPOCHS,
        batch_size=model.BATCH_SIZE,
        callbacks=[model.checkpoint_callback],
        verbose=1,
    )

    print("\n모델 학습 완료")

    # 저장
    vae.save(str(model.VAE_SAVE_PATH))
    encoder.save(str(model.ENC_SAVE_PATH))
    decoder.save(str(model.DEC_SAVE_PATH))

    print("\n[SAVED]")
    print(f" VAE     : {model.VAE_SAVE_PATH}")
    print(f" Encoder : {model.ENC_SAVE_PATH}")
    print(f" Decoder : {model.DEC_SAVE_PATH}")