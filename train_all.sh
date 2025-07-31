#!/bin/bash

# ===============================================
# 1. 여기에 학습시킬 데이터셋 태그들을 입력하세요.
# ===============================================
TAGS=(
    "Electronic_Dance_SynthPop"
    "Rock_Pop"
    "Jazz_Blues"
    "Country"
    "RnB_Soul_Funk_Disco"
    "Classical"
)

# 현재 스크립트의 경로를 기준으로 model.py 경로 설정
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MODEL_PY_PATH="$SCRIPT_DIR/model/model.py"

echo "전체 학습을 시작합니다..."
echo "대상 파일: $MODEL_PY_PATH"
echo "==============================================="

# 각 태그에 대해 반복 실행
for TAG in "${TAGS[@]}"
do
    echo ""
    echo ">>>>> [시작] 데이터셋 태그: $TAG <<<<<"
    echo "==============================================="

    # 2. model.py 파일의 DATASET_TAG 값을 현재 태그로 변경
    # sed 명령어를 사용해 특정 라인을 찾아 교체합니다.
    sed -i "s/DATASET_TAG = .*/DATASET_TAG = \"$TAG\"/" "$MODEL_PY_PATH"
    echo "'$MODEL_PY_PATH'의 DATASET_TAG를 '$TAG'(으)로 변경했습니다."

    # 3. train.py 스크립트 실행
    echo "train.py를 실행합니다..."
    python "$SCRIPT_DIR/model/train.py"

    echo ""
    echo ">>>>> [완료] 데이터셋 태그: $TAG <<<<<"
    echo "==============================================="
    sleep 5 # 다음 학습 전 5초 대기
done

echo "모든 학습이 완료되었습니다."