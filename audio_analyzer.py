import whisper
import librosa
import numpy as np
import os
from google import genai
import json
from datetime import datetime
import joblib
from dotenv import load_dotenv
load_dotenv()
# завантажуємо .env

class PitchAnalyzer:
    def __init__(self, whisper_model_size="small"):
        print(f"Завантаження моделей ({whisper_model_size})... Зачекайте.")

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError(" GEMINI_API_KEY не знайдено у .env")

        self.stt_model = whisper.load_model(whisper_model_size)
        self.client = genai.Client(api_key=gemini_api_key)
        self.target_model = "models/gemini-3-flash-preview"

        try:
            self.voice_model = joblib.load("voice_confidence_model.pkl")
            self.scaler = joblib.load("scaler.pkl")
            print("Власна модель (RandomForest) завантажена успішно.")
        except Exception as e:
            print(f"Помилка завантаження нейронки: {e}")
            self.voice_model = None

        self.filler_words = [
            "ну","типу","якби","е-е","коротше","ось","власне",
            "значить","так би мовити","мм","и-и","е-е-е",
            "тіпа","ті па","like"
        ]

    def extract_voice_features(self, y, sr):
        """Метод витягування ознак (має збігатися з train_voice_model.py)"""
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        return np.hstack([mfccs, chroma, mel, rms, centroid])

    def analyze_content_with_gemini(self, text, target_topic, style, voice_confidence):
        confidence_desc = f"{round(voice_confidence * 100)}%"

        prompt = f"""
        Ти — професійний тренер з публічних виступів. Відповідай тією мовою, якою записане аудіо.



        КОНТЕКСТ АНАЛІЗУ:

        Тема: "{target_topic}"

        Стиль: "{style}"

        Впевненість голосу (визначена твоєю нейронкою): {confidence_desc}



        ЗАВДАННЯ:

    1. Проаналізуй зміст виступу.
    2. Текст отримано через систему розпізнавання мовлення, тому спочатку ВИПРАВ очевидні помилки розпізнавання (наприклад, терміни, випадкові букви у словах).
    НЕ виправляй саму структуру тексту. Не пиши у відповідь виправлений текст. Давай свій остаточний відгук на уже виправлений текст (невеликі виправлення, які необхідні через неточності розпізнавання мовлення)
    3. Оціни зміст за 100-бальною шкалою (topic_score, style_score, structure_score).
    4. У полі feedback дай розгорнуту оцінку виступу. ПОТІМ додатково згадай про впевненість голосу {confidence_desc}. Якщо він низький (менше 50%), поясни, що саме в акустиці могло це спричинити (темп, тремтіння, гучність). (Високою впевненістю є показник більше 60%)
    Проте у відгуку не потрібно прямо писати відсоток впевненості користувача та відсотки(50% та 60%), які є межами визначення якості.  

        Текст для аналізу:
        {text}

        Відповідай СТРОГО у форматі JSON:
        {{
          "topic_score": int,
          "style_score": int,
          "structure_score": int,
          "feedback": "твій відгук",
          "tips": "поради"
        }}
        """

        try:
            response = self.client.models.generate_content(
                model=self.target_model,
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            return json.loads(response.text)
        except Exception as e:
            return {"error": f"Помилка Gemini: {str(e)}"}

    def process_audio(self, audio_path, target_topic, style):
        if not os.path.exists(audio_path):
            return {"error": "Файл не знайдено"}

        print(f"Починаю аналіз: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration > 65:
            return {
                "error": "Аудіо занадто довге",
                "message": f"Ваш запис триває {round(duration)} сек. Будь ласка, обмежте виступ 1 хвилиною для коректного аналізу."
            }

        # --- 1. ТРАНСКРИПЦІЯ (потрібна спочатку для розрахунку WPM) ---
        print("Виконую транскрипцію (Whisper)...")
        context = f"Тема: {target_topic}. Мікропластик, спектроскопія, FTIR, Раман."
        result = self.stt_model.transcribe(audio_path, language=None, initial_prompt=context, fp16=False)
        text = result['text']
        words = text.lower().replace(",", "").replace(".", "").split()
        wpm = (len(words) / duration) * 60 if duration > 0 else 0

        # --- 2. АНАЛІЗ ВПЕВНЕНОСТІ (RandomForest + Оптимізована логіка) ---
        voice_score = 0.5
        if self.voice_model:
            features = self.extract_voice_features(y, sr)
            features_scaled = self.scaler.transform([features])
            probs = self.voice_model.predict_proba(features_scaled)[0]

            # ПІДНЯТА ВАГА: Спокійний голос (probs[1]) тепер дає 0.8 балів
            nn_score = (probs[1] * 0.8) + (probs[2] * 1.0)

            # ОПТИМІЗОВАНА ТИША (top_db=38 робить алгоритм менш суворим)
            intervals = librosa.effects.split(y, top_db=38)
            speaking_time = sum([end - start for start, end in intervals]) / sr
            silence_ratio = (duration - speaking_time) / duration if duration > 0 else 0

            # М'який множник тиші: не знижує бал, якщо тиші менше 40%
            if silence_ratio < 0.40:
                silence_multiplier = 1.0
            else:
                silence_multiplier = max(0.75, 1.15 - silence_ratio)

            # Додаємо бонус за гарний темп мовлення (енергійність)
            wpm_bonus = min(0.08, (wpm / 1500))

            voice_score = (nn_score * silence_multiplier) + wpm_bonus
            voice_score = max(0.15, min(0.98, voice_score))

        # --- 3. ТЕХНІЧНІ МЕТРИКИ ---
        found_fillers = [w for w in words if w in self.filler_words]
        silence_percent = (silence_ratio * 100) if 'silence_ratio' in locals() else 0

        # --- 4. AI АНАЛІЗ ---
        print("Надсилаю на AI-аналіз...")
        ai_review = self.analyze_content_with_gemini(text, target_topic, style, voice_score)

        label = "Низька"
        if voice_score > 0.65:
            label = "Висока"
        elif voice_score > 0.35:
            label = "Середня"

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transcription": text,
            "voice_confidence": {
                "score": round(voice_score, 2),
                "label": label
            },
            "technical_metrics": {
                "duration_sec": round(duration, 1),
                "wpm": round(wpm, 1),
                "fillers_count": len(found_fillers),
                "fillers_list": list(set(found_fillers)),
                "silence_percent": round(silence_percent, 1)
            },
            "ai_analysis": ai_review
        }


if __name__ == "__main__":
    analyzer = PitchAnalyzer(whisper_model_size="small")

    file_name = "Presentation.wav"
    target_topic = "Slang in Social Media as a Means of Shaping Unrealistic Perceptions of Human Appearance"
    style = "Project presentation"

    result = analyzer.process_audio(file_name, target_topic, style)

    print("\n--- РЕЗУЛЬТАТ АНАЛІЗУ ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    with open("last_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)