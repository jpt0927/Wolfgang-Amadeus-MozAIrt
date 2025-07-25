// src/app/page.tsx
"use client";

import { useState, useEffect, FormEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from 'next/image';
import LoadingAnimation from "@/components/LoadingAnimation";
import TonePlayer from "@/components/TonePlayer"; // 새로 만든 TonePlayer를 import

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
];

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [prompt, setPrompt] = useState("");
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

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!prompt || isLoading) return;

    setMusicUrl(null);
    setIsLoading(true);

    setTimeout(() => {
      // --- 2. 시뮬레이션: 랜덤 장르 번호 설정 ---
      // 실제로는 백엔드에서 받은 장르 번호를 여기에 설정합니다.
      const randomGenre = Math.floor(Math.random() * 6);
      setGenre(randomGenre);

      setMusicUrl("https://bitmidi.com/uploads/14266.mid");
      setIsLoading(false);
    }, 5000);
  };

  return (
    <main className="flex min-h-screen w-full items-center justify-center p-4">
      <div className="w-full max-w-md" id="interactive-area">
      <div className="w-full max-w-md">
        {/* 상단 카드는 수정 없음 */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl shadow-lg p-8 border border-white/10">
        <div className="flex items-center gap-4 mb-4">
          <Image
              src="/music-icon.png" // public 폴더의 이미지 경로
              alt="음악 아이콘"
              width={60} // 아이콘 크기 (글자 높이와 비슷하게)
              height={60}
              className="flex-shrink-0 rounded-md relative -top-2" // 창이 줄어들어도 아이콘이 찌그러지지 않도록 설정
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
                      <h2 className="text-xl font-bold text-white mb-4">AI의 창작물</h2>
                      <TonePlayer url={musicUrl} genre={genre}/>
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