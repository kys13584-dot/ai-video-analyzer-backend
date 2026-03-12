import os
import subprocess
import tempfile
import librosa # type: ignore
import numpy as np # type: ignore
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

# ── Whisper worker (runs in a separate process) ──────────────────────────────

def _transcribe_in_subprocess(audio_path: str) -> dict:
    """
    Runs Whisper transcription in isolation.
    Executed in a separate process by ProcessPoolExecutor so it cannot
    block the FastAPI event loop or other threads.

    1단계: 언어 자동 감지 (detect_language)
    2단계: 감지된 언어로 고정하여 전사 (혼합 언어 출력 방지)
    """
    import whisper as _w  # type: ignore

    model = _w.load_model("tiny")

    # 1단계: 언어 감지
    audio = _w.load_audio(audio_path)
    audio_clip = _w.pad_or_trim(audio)
    mel = _w.log_mel_spectrogram(audio_clip).to(model.device)
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)  # type: ignore[arg-type]
    confidence = float(probs[detected_lang]) * 100  # type: ignore[index]
    print(f"[Whisper] Detected language: {detected_lang} (confidence: {confidence:.1f}%)")

    # 2단계: 감지된 언어로 고정하여 전사
    return model.transcribe(audio_path, fp16=False, language=detected_lang)


# ── Main analyzer class ──────────────────────────────────────────────────────

class AudioAnalyzer:
    """
    Extracts speech, pacing, energy, and tone features from video audio.

    Whisper is run in a subprocess with a hard 90-second timeout so the
    server never gets permanently blocked by a slow transcription.
    """

    WHISPER_TIMEOUT = 90  # seconds

    def analyze_audio(self, video_path: str) -> dict:
        result = {
            "speech_tempo": 0.0,
            "has_music": False,
            "emotional_tone": "neutral",
            "pacing": 0.0,
            "subtitle_density": 0.0,
            "audio_energy": 0.0,
            "transcript": "",
        }

        if not os.path.exists(video_path):
            print(f"[Audio] File not found: {video_path}")
            return result

        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            # ── Step 1: Extract audio via ffmpeg ──────────────────────────
            cmd = [
                "ffmpeg", "-i", video_path,
                "-q:a", "0", "-map", "a",
                "-ar", "16000", "-ac", "1",
                temp_audio, "-y",
            ]
            subprocess.run(cmd, capture_output=True, check=False, timeout=30)

            if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
                print("[Audio] No audio track found, skipping STT")
                return result

            # ── Step 2: Whisper STT in separate process with timeout ───────
            print("[Audio] Starting Whisper (subprocess, 90s timeout)...")
            try:
                with ProcessPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_transcribe_in_subprocess, temp_audio)
                    stt_result = future.result(timeout=self.WHISPER_TIMEOUT)

                text: str = stt_result.get("text", "")
                segments: list = stt_result.get("segments", [])
                result["transcript"] = text.strip()
                print(f"[Audio] Whisper done. Words: {len(text.split())}")

                # Speech tempo & subtitle density
                total_speech_duration = sum(
                    seg["end"] - seg["start"] for seg in segments
                )
                word_count = len(text.split())
                if total_speech_duration > 0:
                    wpm = (word_count / total_speech_duration) * 60
                    result["speech_tempo"] = wpm
                    result["subtitle_density"] = min(wpm / 200.0, 1.0)
                    result["pacing"] = min(wpm / 150.0, 2.0)

            except FuturesTimeout:
                print(f"[Audio] Whisper timed out after {self.WHISPER_TIMEOUT}s — using defaults")
            except Exception as whisper_err:
                print(f"[Audio] Whisper error: {whisper_err} — using defaults")

            # ── Step 3: Audio energy & tone via librosa ───────────────────
            try:
                y, sr = librosa.load(temp_audio, sr=16000)
                if len(y) > 0:
                    rms = librosa.feature.rms(y=y)[0]
                    mean_energy = float(np.mean(rms))
                    variance_energy = float(np.var(rms))

                    result["audio_energy"] = min(mean_energy * 10, 1.0)

                    pacing_val = float(result["pacing"])
                    if mean_energy > 0.1 and pacing_val > 1.2:
                        result["emotional_tone"] = "energetic"
                    elif variance_energy > 0.05:
                        result["emotional_tone"] = "dramatic"
                    elif pacing_val < 0.8:
                        result["emotional_tone"] = "calm"
                    else:
                        result["emotional_tone"] = "informative"

                    if variance_energy < 0.01 and mean_energy > 0.01:
                        result["has_music"] = True
            except Exception as librosa_err:
                print(f"[Audio] librosa error: {librosa_err}")

        except Exception as e:
            print(f"[Audio] Unexpected error: {e}")
        finally:
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except Exception:
                    pass

        return result
