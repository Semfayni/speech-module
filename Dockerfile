# Використовуємо образ з Python 3.10
FROM python:3.10-slim

# Встановлюємо ffmpeg та системні залежності для librosa
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо залежності та встановлюємо їх
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо всі файли проєкту (включаючи .pkl моделі)
COPY . .

# Запускаємо через gunicorn.
# Збільшуємо timeout, бо Whisper та Gemini можуть думати довго
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 main:app