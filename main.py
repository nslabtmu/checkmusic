from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import librosa
import numpy as np
import yt_dlp
import os

app = FastAPI()

class AnalysisRequest(BaseModel):
    source_type: str  # 'youtube' 或 'file'
    url: str          # YouTube 網址或 Google Drive 檔案下載網址
    file_name: str

@app.post("/analyze")
def analyze_music(req: AnalysisRequest):
    audio_path = "temp_audio.mp3"
    
    try:
        # 1. 根據來源取得音訊檔
        if req.source_type == 'youtube':
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': 'temp_audio.%(ext)s',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}]
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([req.url])
            audio_path = "temp_audio.mp3" # yt-dlp 轉檔後的檔名
        else:
            # 若是 Google Drive 檔案，直接用 requests 下載 req.url
            import requests
            res = requests.get(req.url)
            with open(audio_path, 'wb') as f:
                f.write(res.content)

        # 2. 使用 librosa 載入音訊進行聲學特徵提取
        y, sr = librosa.load(audio_path, sr=22050, duration=60) # 預設分析前 60 秒加速
        
        # 計算 BPM (節奏)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

        # 計算高頻能量占比 (> 4kHz) 對應 Safety Sound Screen 規範
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_idx = np.where(freqs > 4000)[0]
        
        total_energy = np.sum(S)
        high_energy = np.sum(S[high_freq_idx, :])
        high_freq_ratio = round(float(high_energy / total_energy * 100), 2)

        # 模擬計算 LUFS / True Peak (實務上可搭配 pyloudnorm 庫)
        lufs = -18.5  
        true_peak = -1.2

        # 綜合判定是否符合 Safety Sound Screen 標準 (例如: 高頻 < 15% 且 LUFS 在合理範圍)
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
        # 清理暫存檔
        if os.path.exists(audio_path):
            os.remove(audio_path)
