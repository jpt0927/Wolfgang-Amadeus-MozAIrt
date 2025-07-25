// src/components/ClickFX.tsx
"use client";

import { useState, useEffect, useCallback } from 'react';

// 클릭 시 나타날 음표 종류
const musicalNotes = ['🎵', '🎶', '♪', '♫', '🎼', '♭', '♯'];

// 각 음표의 타입을 정의합니다.
interface Note {
  id: number;
  x: number;
  y: number;
  char: string;
}

export default function ClickFX() {
  const [notes, setNotes] = useState<Note[]>([]);

  const handleClick = useCallback((e: MouseEvent) => {
    // 클릭된 대상(e.target)이 'interactive-area' 또는 그 내부에 있는지 확인
    if ((e.target as HTMLElement).closest('#interactive-area')) {
      return; // 내부에 있다면, 여기서 함수를 종료 (음표를 만들지 않음)
    }

    // 새 음표 객체를 만듭니다.
    const newNote: Note = {
      id: Date.now() + Math.random(),
      x: e.clientX,
      y: e.clientY,
      char: musicalNotes[Math.floor(Math.random() * musicalNotes.length)],
    };

    setNotes((prev) => [...prev, newNote]);

    setTimeout(() => {
      setNotes((prev) => prev.filter((note) => note.id !== newNote.id));
    }, 2000);
  }, []);

  // 컴포넌트가 로드될 때 클릭 이벤트 리스너를 추가하고, 사라질 때 제거합니다.
  useEffect(() => {
    window.addEventListener('click', handleClick);
    return () => {
      window.removeEventListener('click', handleClick);
    };
  }, [handleClick]);

  // 화면에 음표들을 렌더링합니다.
  return (
    <>
      {notes.map(({ id, x, y, char }) => (
        <span
          key={id}
          className="music-note-fx"
          style={{
            left: `${x}px`,
            top: `${y}px`,
          }}
        >
          {char}
        </span>
      ))}
    </>
  );
}