// src/app/page.tsx
"use client";

import { useState, useEffect, FormEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from 'next/image';
import LoadingAnimation from "@/components/LoadingAnimation";
import TonePlayer from "@/components/TonePlayer";


const loadingPhrases = [
  "악상을 펼치는 중...",
  "멜로디를 구성하는 중...",
  "영감이 떠오르는 중...",
  "구자윤이 롤 하는 중...",
  "코드를 조합하는 중...",
  "음악을 완성하는 중...",
];

const genreNames = [
  "Country",
  "Classical",
  "Electronic & Dance",
  "Jazz & Blues",
  "RnB & Soul & Funk",
  "Rock & Pop",
  "Fantastic"
];

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [title, setTitle] = useState("AI의 음악");
  const [musicUrl, setMusicUrl] = useState<string | null>(null);
  const [currentPhrase, setCurrentPhrase] = useState(loadingPhrases[0]);
  const [genre, setGenre] = useState(1);

  useEffect(() => {
    if (isLoading) {
      const interval = setInterval(() => {
        setCurrentPhrase((prev) => loadingPhrases[(loadingPhrases.indexOf(prev) + 1) % loadingPhrases.length]);
      }, 2500);
      return () => clearInterval(interval);
    }
  }, [isLoading]);

  // ############ 여기가 핵심 수정 부분 ############
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!prompt || isLoading) return;

    setMusicUrl(null);
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt }),
      });

      if (!response.ok) {
        throw new Error('API 요청에 실패했습니다.');
      }

      // 1. 헤더에서 제목과 장르 정보 추출
      const headerTitle = response.headers.get('X-Music-Title');
      const headerGenre = response.headers.get('X-Music-Genre');

      // 2. 응답 본문에서 MIDI 파일(blob) 데이터 추출
      const blob = await response.blob();
      
      // 3. Blob 데이터로 브라우저에서만 사용 가능한 임시 URL 생성
      const newMusicUrl = URL.createObjectURL(blob);
      
      // 4. 상태 업데이트
      // 백엔드에서 URL 인코딩된 제목을 디코딩
      setTitle(headerTitle ? decodeURIComponent(headerTitle) : "제목 없음");
      setGenre(headerGenre ? parseInt(headerGenre, 10) : 1);
      setMusicUrl(newMusicUrl);

      console.log("음악 생성 완료:", {
        title: headerTitle ? decodeURIComponent(headerTitle) : "제목 없음",
        genre: headerGenre,
        musicUrl: newMusicUrl
      });

    } catch (error) {
      console.error("음악 생성 실패:", error);
    } finally {
      setIsLoading(false);
    }
  };
  // ############ 여기까지 핵심 수정 부분 ############

  // 컴포넌트의 return 부분은 수정할 필요 없습니다.
  return (
    <main className="flex min-h-screen w-full items-center justify-center p-4">
      <div className="w-full max-w-md" id="interactive-area">
      <div className="w-full max-w-md">
        {/* 상단 카드 */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl shadow-lg p-8 border border-white/10">
        <div className="flex items-center gap-4 mb-4">
          <Image
              src="/music-icon.png"
              alt="음악 아이콘"
              width={60}
              height={60}
              className="flex-shrink-0 rounded-md relative -top-2"
            />
          <h1 className="text-2xl font-bold text-white leading-snug mb-4">
            어떤 분위기의 음악을<br />만들어 드릴까요?
          </h1>
          </div>
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="예: 비 내리는 밤, 재즈 바의 쓸쓸한 피아노"
              disabled={isLoading}
              className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg placeholder-gray-400 focus:ring-2 focus:ring-indigo-400 focus:outline-none transition-all disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isLoading || !prompt}
              className="w-full mt-4 px-4 py-3 bg-indigo-500 text-white text-base font-bold rounded-lg hover:bg-indigo-600 transition-colors disabled:bg-gray-500/50 disabled:cursor-not-allowed"
            >
              음악 생성하기
            </button>
          </form>
        </div>

        {/* 로딩 또는 결과 카드 */}
        <div className="relative h-56 mt-4">
          <AnimatePresence>
            {(isLoading || musicUrl) && (
              <motion.div
                key={isLoading ? 'loader' : 'player'}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="absolute w-full bg-white/10 backdrop-blur-md rounded-xl shadow-lg p-8 flex flex-col items-center justify-center min-h-[14rem] border border-white/10"
              >
                {isLoading ? (
                  <div className="text-center">
                    <LoadingAnimation />
                    <AnimatePresence mode="wait">
                      <motion.p
                        key={currentPhrase}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.5 }}
                        className="text-lg text-gray-300 font-medium mt-4"
                      >
                        {currentPhrase}
                      </motion.p>
                    </AnimatePresence>
                  </div>
                ) : (
                  musicUrl && (
                    <div className="w-full text-center">
                      <h2 className="text-xl font-bold text-white mb-4">{title}</h2>
                      <TonePlayer url={musicUrl} genre={genre} title={title}/>
                      <p className="text-sm text-gray-400 mt-3">
                        AI가 제안하는 장르: {genreNames[genre]}
                      </p>
                    </div>
                  )
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
      </div>
    </main>
  );
}