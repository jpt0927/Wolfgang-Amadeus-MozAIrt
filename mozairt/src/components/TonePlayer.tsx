// src/components/TonePlayer.tsx
"use client";

import { useEffect, useRef, useState } from 'react';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';
import { audioBufferToWav } from '@/utils/wavEncoder';

// 아이콘 SVG 컴포넌트
const PlayIcon = () => ( <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg> );
const PauseIcon = () => ( <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg> );

const createInstrumentByGenre = (genre: number) => {
  switch (genre) {
    case 0: // Country
      return new Tone.PluckSynth({ attackNoise: 0.2, dampening: 5000, resonance: 0.7, release: 2.5 });
    case 2: // Electronic
      return new Tone.PolySynth(Tone.Synth, { oscillator: { type: 'fatsawtooth' }, envelope: { attack: 0.2, decay: 0.1, sustain: 0.6, release: 1.5 } });
    case 3: case 4: // Jazz & RnB
      return new Tone.PolySynth(Tone.FMSynth, { harmonicity: 4, modulationIndex: 10, envelope: { attack: 0.01, decay: 0.2, sustain: 0.3, release: 1 }, modulationEnvelope: { attack: 0.02, decay: 0.2, sustain: 0.2, release: 0.4 } });
    case 5: // Rock
      const dist = new Tone.Distortion(0.6);
      const eq = new Tone.EQ3({ low: -2, mid: 4, high: -2 });
      const reverb = new Tone.Reverb({ decay: 1.2, wet: 0.2 });
      const synth = new Tone.PolySynth(Tone.Synth, { oscillator: { type: 'fatsawtooth', count: 3, spread: 20 }, envelope: { attack: 0.01, decay: 0.3, sustain: 0.2, release: 0.5 } });
      // synth.chain의 마지막에 Tone.Destination을 연결해줘야 소리가 납니다.
      synth.chain(dist, eq, reverb, Tone.Destination);
      return synth;
    case 6: // Fantastic
    case 1: default: // Classical
      return new Tone.Sampler({ urls: {
        A0: "A0.mp3",
        C1: "C1.mp3",
        "D#1": "Ds1.mp3",
        "F#1": "Fs1.mp3",
        A1: "A1.mp3",
        C2: "C2.mp3",
        "D#2": "Ds2.mp3",
        "F#2": "Fs2.mp3",
        A2: "A2.mp3",
        C3: "C3.mp3",
        "D#3": "Ds3.mp3",
        "F#3": "Fs3.mp3",
        A3: "A3.mp3",
        C4: "C4.mp3",
        "D#4": "Ds4.mp3",
        "F#4": "Fs4.mp3",
        A4: "A4.mp3",
        C5: "C5.mp3",
        "D#5": "Ds5.mp3",
        "F#5": "Fs5.mp3",
        A5: "A5.mp3",
        C6: "C6.mp3",
        "D#6": "Ds6.mp3",
        "F#6": "Fs6.mp3",
        A6: "A6.mp3",
        C7: "C7.mp3",
        "D#7": "Ds7.mp3",
        "F#7": "Fs7.mp3",
        A7: "A7.mp3",
        C8: "C8.mp3"
      }, baseUrl: "https://tonejs.github.io/audio/salamander/", release: 1 });
  }
};

type TonePlayerProps = {
  url: string;
  genre: number;
  title: string;
};

const TonePlayer = ({ url, genre, title }: TonePlayerProps) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isRendering, setIsRendering] = useState(false);

  const instrumentRef = useRef<any>(null);
  const midiRef = useRef<Midi | null>(null);

  useEffect(() => {
    let instrument: any;
    const loadMidi = async () => {
      try {
        setIsLoaded(false);
        instrument = createInstrumentByGenre(genre);

        // Rock 장르(case 5)는 이미 Destination에 연결되어 있으므로 중복 연결하지 않습니다.
        if (genre !== 5) {
          instrument.toDestination();
        }
        instrumentRef.current = instrument;
        
        if (instrument instanceof Tone.Sampler) await Tone.loaded();

        const midi = await Midi.fromUrl(url);
        midiRef.current = midi;
        setDuration(midi.duration);

        midi.tracks.forEach(track => {
          track.notes.forEach(note => {
            Tone.Transport.schedule(time => {
              instrumentRef.current?.triggerAttackRelease(note.name, note.duration, time, note.velocity);
            }, note.time);
          });
        });
        setIsLoaded(true);
      } catch (error) { console.error("로드 실패:", error); setIsLoaded(false); }
    };
    if (url) loadMidi();
    
    return () => {
      Tone.Transport.stop();
      Tone.Transport.cancel();
      instrumentRef.current?.dispose();
      setIsPlaying(false);
      setCurrentTime(0);
    };
  }, [url, genre]);
  
  const handleDownload = async (title: string) => {
    if (!isLoaded || isRendering || !midiRef.current) return;
    setIsRendering(true);
    try {
      const buffer = await Tone.Offline(async (offlineContext) => {
        const instrument = createInstrumentByGenre(genre);
        // Rock 장르가 아닌 경우에만 오프라인 컨텍스트의 destination에 연결
        if (genre !== 5) {
          instrument.connect(offlineContext.destination);
        }

        if (instrument instanceof Tone.Sampler) await Tone.loaded();
        
        midiRef.current?.tracks.forEach(track => {
          track.notes.forEach(note => {
            offlineContext.transport.schedule(time => {
                instrument.triggerAttackRelease(note.name, note.duration, time, note.velocity);
            }, note.time)
          });
        });
        offlineContext.transport.start();
      }, duration);

      const wavBlob = audioBufferToWav(buffer);
      const blobUrl = URL.createObjectURL(wavBlob);
      const anchor = document.createElement('a');
      anchor.href = blobUrl;
      anchor.download = `${title}.wav`;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error("오디오 파일 생성 실패:", error);
    } finally {
      setIsRendering(false);
    }
  };
  
  const handlePlayPause = async () => {
    if (!isLoaded) return;
    if (Tone.context.state !== 'running') await Tone.context.resume();
    setIsPlaying(state => !state);
  };

  useEffect(() => {
    if (isPlaying) Tone.Transport.start();
    else Tone.Transport.pause();
  }, [isPlaying]);

  useEffect(() => {
    const timeUpdateInterval = setInterval(() => {
      if (Tone.Transport.state === 'started') {
        const newTime = Tone.Transport.seconds;
        setCurrentTime(newTime);
        if (newTime >= duration) {
          setIsPlaying(false);
          Tone.Transport.stop();
          setCurrentTime(0); // 재생이 끝나면 처음으로
        }
      }
    }, 100);
    return () => clearInterval(timeUpdateInterval);
  }, [duration]);

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!isLoaded) return;
    const newTime = parseFloat(e.target.value);
    setCurrentTime(newTime);
    Tone.Transport.seconds = newTime;
  };

  const formatTime = (seconds: number) => {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${min}:${sec}`;
  };

  return (
    <div className="w-full flex flex-col items-center gap-4">
      <div className="w-full flex items-center gap-4 p-4 bg-white/5 rounded-lg border border-white/10">
        <button onClick={handlePlayPause} disabled={!isLoaded} className="flex items-center justify-center w-12 h-12 bg-indigo-500 text-white rounded-full flex-shrink-0 hover:bg-indigo-600 transition-colors disabled:bg-gray-500/50 disabled:cursor-not-allowed">
          {isLoaded ? (isPlaying ? <PauseIcon /> : <PlayIcon />) : '...'}
        </button>
        <div className="w-full flex flex-col justify-center">
          <input type="range" min="0" max={duration} value={currentTime} onChange={handleSeek} disabled={!isLoaded} className="w-full h-1.5 bg-white/20 rounded-lg appearance-none cursor-pointer" style={{ background: `linear-gradient(to right, #818cf8 ${ (currentTime / duration) * 100 }%, #4b5563 ${ (currentTime / duration) * 100 }%)`}} />
          <div className="flex justify-between text-xs text-gray-400 mt-1.5">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>
      </div>
      <button
        onClick={() => handleDownload(title)}
        disabled={!isLoaded || isRendering}
        className="w-full px-4 py-3 border border-gray-500 text-gray-300 text-base font-bold rounded-lg hover:bg-white/10 transition-colors disabled:bg-gray-500/50 disabled:cursor-not-allowed"
      >
        {isRendering ? 'WAV 파일 생성 중...' : 'WAV 파일로 저장'}
      </button>
    </div>
  );
};

export default TonePlayer;