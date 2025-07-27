# train.py

# model.py에서 필요한 코드 임포트
import model  # model.py 파일을 임포트

if __name__ == "__main__":
    # 모델 학습 시작
    print("모델 학습 시작...")

    # 모델 학습 실행
    model.vae.fit(model.data, model.data, 
                  epochs=model.EPOCHS, 
                  batch_size=model.BATCH_SIZE, 
                  callbacks=[model.checkpoint_callback, model.generate_sample_callback])

    print("모델 학습 완료")
