# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # CORS를 위한 미들웨어
from pydantic import BaseModel
import time
import random

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정: 프론트엔드(Next.js) 서버からの요청을 허용
origins = [
    "http://localhost:3000", # Next.js 개발 서버 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # 모든 HTTP 메소드 허용
    allow_headers=["*"], # 모든 HTTP 헤더 허용
)

# 요청 본문(body)의 타입을 정의
class PromptRequest(BaseModel):
    prompt: str

# /api/generate 주소로 POST 요청을 받는 API 엔드포인트
@app.post("/api/generate")
async def generate_music(request: PromptRequest):
    print(f"음악 생성 요청 프롬프트: {request.prompt}")

    # 여기에 Ollama와 VAE 모델을 호출하는 로직이 들어갑니다.
    # 지금은 시뮬레이션을 위해 5초 대기
    time.sleep(5)

    # 모델 처리 결과
    music_url = "https://bitmidi.com/uploads/14266.mid"
    genre = random.randint(0, 5)

    return {"musicUrl": music_url, "genre": genre}