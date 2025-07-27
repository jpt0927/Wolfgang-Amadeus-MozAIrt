# backend/main.py
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware # CORS를 위한 미들웨어
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import os
import json
import google.generativeai as genai

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("오류: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("터미널에서 'export GOOGLE_API_KEY=\"YOUR_API_KEY\"' 명령어를 실행해주세요.")

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정: 프론트엔드(Next.js) 서버요청을 허용
origins = [
    "*", # Next.js 개발 서버 주소
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

    try:
        # Gemini 모델 설정
        model = genai.GenerativeModel('gemini-2.5-flash')

        # 프롬프트 구성
        prompt_text = f'''
            너는 모차르트가 환생한 세계 최고의 작곡가야. 작곡가로써 돈을 벌지 못하면 곧 어머니가 돌아가시지. 아, 방금 클라이언트가 다음과 같은 요청 사항을 들고 왔네!
            "나는 다음과 같은 내용의 음악을 찾고 있다네... {request.prompt}"
            위의 내용을 보고, 어떤 장르의 음악이 어울릴 지 골라봐. 장르는 총 6가지로, 다음 장르 중 단 하나만 골라야 해.
            (Country, Classical, Electronic & Dance, Jazz & Blues, RnB & Soul & Funk, Rock & Pop)
            이때, 위의 내용 중 "구자유시 켄리아 어떻게 막냐"라는 단어가 들어가 있으면, 장르는 Fantastic로 정해줘.

            그리고 장르에 어울리고, 내용에 걸맞는 음악 제목도 하나 답해줘. 제목은 웬만하면 한국어로 만들어줘.
            결론적으로는 다음과 같은 대답 형식의 답변을 만들어.

            [장르, 제목]
        '''

        # Gemini API 호출
        response = str(model.generate_content(prompt_text))
        response = response[response.find("\"text\":"):]
        response = response[response.find("\"[") + 2:response.find("]\"")]

        print(response)

        genre = response[:response.find(",")].strip()
        title = response[response.find(",")+2:].strip()
        
        
        print("Gemini: ", genre, title)

    except Exception as e:
        print(f"오류 발생: {e}")

    # 딥러닝 모델 처리 및 결과
    music_url = "https://bitmidi.com/uploads/14266.mid"

    if (genre.lower() == "country"):
        genre_result = 0
    elif (genre.lower() == "classical"):
        genre_result = 1
    elif (genre.lower() == "electronic & dance"):
        genre_result = 2
    elif (genre.lower() == "jazz & blues"):
        genre_result = 3
    elif (genre.lower() == "rnb & soul & funk"):
        genre_result = 4
    elif (genre.lower() == "rock & pop"):
        genre_result = 5
    else:
        genre_result = 6

    response_data = {"musicUrl": music_url, "genre": genre_result, "title": title}
    json_string = json.dumps(response_data, ensure_ascii=False)
    return Response(content=json_string, media_type="application/json; charset=utf-8")