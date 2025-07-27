# train.py
import model

if __name__ == "__main__":
    print("모델 학습 시작...")

    history = model.vae.fit(
        model.data, model.data,
        epochs=model.EPOCHS,
        batch_size=model.BATCH_SIZE,
        callbacks=[model.checkpoint_callback, model.generate_sample_callback],
        verbose=1,
    )

    print("모델 학습 완료")

    # 최종 저장 (model.py의 __main__에서는 저장되지만,
    # train.py 경로로 학습할 땐 여기서 저장해야 함)
    model.vae.save(model.VAE_SAVE_PATH)
    model.encoder.save(model.ENC_SAVE_PATH)
    model.decoder.save(model.DEC_SAVE_PATH)

    print("[SAVED]")
    print("  VAE     :", model.VAE_SAVE_PATH)
    print("  Encoder :", model.ENC_SAVE_PATH)
    print("  Decoder :", model.DEC_SAVE_PATH)
