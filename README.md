# Music VAE 프로젝트 : Wolfgang Amadeus MozAIrt

**변분 오토인코더(VAE)** 를 기반으로 **클래식 음악** 피아노 롤을 학습 → 새로운 음악을 생성하는 모델입니다.

---

## ✨ 주요 기능
| 기능 | 설명 |
|-----|-----|
| **VAE 기반 음악 생성** | 클래식 피아노 롤 데이터를 학습해 새로운 음악을 만듭니다. |
| **GPU 가속** | TensorFlow GPU로 학습 속도 향상 |
| **체크포인트 저장** | 학습 도중 최적 모델을 주기적으로 저장 |
| **MIDI 출력** | 생성 결과를 MIDI 파일로 저장 (변환 함수 포함) |

---

## 🚀 설치 방법

### 1. 리포지토리 클론
```bash
git clone https://github.com/yourusername/music-vae-project.git
cd music-vae-project
```

### 2. 가상 환경 설정
```bash
# pyenv / pyenv-virtualenv 설치 후:
pyenv install 3.8.12
pyenv virtualenv 3.8.12 music-vae-env
pyenv activate music-vae-env
```

### 3. 의존성 설치 후 TensorFlow GPU 선택
```bash
pip install tensorflow-gpu==2.11
```

## 학습
<img width="1897" height="764" alt="Image" src="https://github.com/user-attachments/assets/ae32a940-68fa-4d6b-8888-8ebe2620a400" />
### 데이터셋

데이터셋은 피아노 롤 형식으로 classic, country, EDM, Jazz, RnB, Rock_Pop 음악을 표현한 것입니다. 피아노 롤은 128개의 피치에 대해 시간별로 키가 눌려졌는지 여부를 나타내는 이진 행렬입니다.

### 모델 아키텍처

모델은 VAE와 LSTM(Long Short-Term Memory) 레이어를 사용하여 음악의 시퀀스적 의존성을 학습합니다. 모델의 구조는 다음과 같습니다:

인코더: 입력 시퀀스를 잠재 공간(latent space) 벡터로 압축합니다.

잠재 공간: 음악 시퀀스의 압축된 표현입니다.

디코더: 잠재 벡터를 입력받아 원래의 음악 시퀀스로 재구성합니다.

후처리: 구성된 음악 시퀀스 데이터를 후처리합니다.

### 학습 절차
재구성 손실: 입력 시퀀스를 재구성하는 오류를 측정합니다.

KL 발산: 잠재 공간 분포가 표준 정규 분포와 얼마나 차이가 나는지 측정합니다.

학습: 모델은 1000 에포크 동안 학습하며, 이 과정에서 VAE 손실 함수를 최적화합니다.


## 기술 스택
<img width="1874" height="817" alt="Image" src="https://github.com/user-attachments/assets/6e108a6b-1352-43b2-b083-a3a0befd5679" />
