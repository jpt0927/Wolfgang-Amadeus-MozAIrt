# backend/main.py
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # FileResponse로 변경
from pydantic import BaseModel
import os
import json
import subprocess # 스크립트 실행을 위해 추가
import urllib.parse # 한글 제목을 헤더에 담기 위해 추가
import google.generativeai as genai

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("오류: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Music-Title", "X-Music-Genre"],
)

# 요청 본문 타입 정의
class PromptRequest(BaseModel):
    prompt: str

# API 엔드포인트
@app.post("/api/generate")
async def generate_music(request: PromptRequest):
    print(f"음악 생성 요청 프롬프트: {request.prompt}")
    
    title = ""
    genre_result = 0

    # 1. Gemini API로 장르와 제목 생성
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt_text = f'''
            너는 모차르트가 환생한 세계 최고의 작곡가야. 작곡가로써 돈을 벌지 못하면 곧 어머니가 돌아가시지. 아, 방금 클라이언트가 다음과 같은 요청 사항을 들고 왔네!
            "나는 다음과 같은 내용의 음악을 찾고 있다네... {request.prompt}"
            위의 내용을 보고, 어떤 장르의 음악이 어울릴 지 골라봐. 장르는 총 6가지로, 다음 장르 중 단 하나만 골라야 해.
            (Country, Classical, Electronic & Dance, Jazz & Blues, RnB & Soul & Funk, Rock & Pop, Fantastic)
            이때, 요청 내용이 애매해서 장르를 정하기 어려우면 장르는 Fantastic로 정해줘.

            그리고 장르에 어울리고, 내용에 걸맞는 음악 제목도 하나 답해줘. 제목은 웬만하면 한국어로 만들어줘.
            결론적으로는 다음과 같은 대답 형식의 답변을 만들어. 꼭 형식을 지켜야 한다.

            [장르, 제목]
        '''
        
        response_text = model.generate_content(prompt_text).text
        response_text = response_text.strip().replace('[', '').replace(']', '')
        
        parts = response_text.split(',')
        genre = parts[0].strip()
        title = parts[1].strip()

        print("Gemini 응답 ->", f"장르: {genre},", f"제목: {title}")
        
        genre_map = {
            "country": 0, "classical": 1, "electronic & dance": 2,
            "jazz & blues": 3, "rnb & soul & funk": 4, "rock & pop": 5, "fantastic": 6
        }
        genre_result = genre_map.get(genre.lower(), 6) # 기본값 Classical

    except Exception as e:
        print(f"Gemini API 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="Gemini API 처리 중 오류가 발생했습니다.")

    # 2. generate_music.py 스크립트 실행하여 MIDI 파일 생성
    try:
        # 실행할 스크립트와 생성될 파일 경로 정의
        # main.py가 backend 폴더 안에 있으므로 경로를 올바르게 지정
        script_path = os.path.join(os.path.dirname(__file__), "generate_music.py")
        output_midi_path = os.path.join(os.path.dirname(__file__), "generated_music.mid")

        print(f"{script_path} 실행 중...")
        
        # `subprocess.run`으로 외부 파이썬 스크립트 실행
        # `check=True`는 스크립트 실행 실패 시 오류를 발생시킴
        process = subprocess.run(
            ["python3", script_path], 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        print("스크립트 실행 완료.")
        
        if not os.path.exists(output_midi_path):
            raise FileNotFoundError("스크립트가 실행되었지만 MIDI 파일이 생성되지 않았습니다.")

    except subprocess.CalledProcessError as e:
        print(f"음악 생성 스크립트 오류: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"음악 파일 생성 스크립트 실행에 실패했습니다.")
    except FileNotFoundError as e:
        print(f"파일 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


    # 3. 생성된 MIDI 파일을 FileResponse로 응답
    # 파일과 함께 추가 정보를 보내기 위해 커스텀 헤더 사용
    headers = {
        # 한글 제목이 깨지지 않도록 URL 인코딩 처리
        "X-Music-Title": urllib.parse.quote(title),
        "X-Music-Genre": str(genre_result)
    }

    return FileResponse(
        path=output_midi_path,
        media_type='audio/midi',
        filename='generated_music.mid', # 다운로드 시 사용될 파일 이름
        headers=headers
    )