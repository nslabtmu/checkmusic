from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import librosa
import numpy as np
import yt_dlp
import os

app = FastAPI()

class AnalysisRequest(BaseModel):
    source_type: str  # 'youtube' 或 'file'
    url: str          # 連結
    file_name: str

@app.post("/analyze")
def analyze_music(req: AnalysisRequest):
    audio_path = "temp_audio.mp3"
    
    try:
        if req.source_type == 'youtube':
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': 'temp_audio.%(ext)s',
                'cookiefile': 'cookies.txt',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192'
                }]
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([req.url])
            audio_path = "temp_audio.mp3"
        else:
            import requests
            res = requests.get(req.url)
            with open(audio_path, 'wb') as f:
                f.write(res.content)

        # 使用 librosa 分析音樂特徵
        y, sr = librosa.load(audio_path, sr=22050, duration=60)
        
        # 1. 計算節奏 (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

        # 2. 計算高頻能量占比 (> 4kHz) - 對應 Safety Sound Screen
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_idx = np.where(freqs > 4000)[0]
        
        total_energy = np.sum(S)
        high_energy = np.sum(S[high_freq_idx, :])
        high_freq_ratio = round(float(high_energy / total_energy * 100), 2)

        # 3. 模擬響度與峰值
        lufs = -18.5  
        true_peak = -1.2

        # 4. 判定是否符合安全標準
        is_safe = high_freq_ratio < 15.0 and true_peak < -1.0

        return {
            "bpm": round(bpm, 1),
            "lufs": lufs,
            "true_peak": true_peak,
            "high_freq_ratio": high_freq_ratio,
            "is_safe": is_safe
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
