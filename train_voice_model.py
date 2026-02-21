import librosa
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # Змінено на RandomForest
import joblib

# 1. ФУНКЦІЯ ВИТЯГУВАННЯ ОЗНАК (залишається розширеною)
def extract_features(file_path):
    # Завантажуємо аудіо
    y, sr = librosa.load(file_path, duration=3)

    # MFCC (Тембр голосу)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    # Chroma (Гармонійна складова)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

    # Mel-спектрограма (Частотний розподіл)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

    # Енергія/Гучність (RMS)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)

    # Спектральний центроїд (Яскравість голосу)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)

    return np.hstack([mfccs, chroma, mel, rms, centroid])


# 2. ПІДГОТОВКА ТА НАВЧАННЯ
def prepare_and_train(dataset_path):
    data = []
    labels = []
    print("Починаю збір ознак з датасету (RandomForest)...")

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    parts = file.split('-')
                    emotion = int(parts[2])

                    # ЛОГІКА ТРЬОХ КЛАСІВ:
                    if emotion in [4, 6]:    # Сум, Страх
                        label = 0  # Низька
                    elif emotion in [1, 2]:  # Нейтральний, Спокій
                        label = 1  # Середня
                    elif emotion in [3, 5]:  # Щастя, Гнів (Енергія)
                        label = 2  # Висока
                    else:
                        continue

                    try:
                        features = extract_features(os.path.join(folder_path, file))
                        data.append(features)
                        labels.append(label)
                    except Exception as e:
                        print(f"Помилка файлу {file}: {e}")

    X = np.array(data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Нормалізація важлива для порівняння фіч
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Навчаю RandomForest на {len(X_train)} зразках...")

    # Створюємо ансамбль із 200 дерев
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,       # Щоб модель не перенавчилася на 100%
        min_samples_leaf=2, # Додає стабільності
        random_state=42,
        n_jobs=-1           # Використовувати всі ядра процесора
    )

    model.fit(X_train, y_train)

    # 4. ЗБЕРЕЖЕННЯ
    joblib.dump(model, "voice_confidence_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    test_acc = model.score(X_test, y_test)
    print(f"--- НАВЧАННЯ ЗАВЕРШЕНО ---")
    print(f"Точність моделі на тестах: {round(test_acc * 100, 1)}%")
    print("Файли оновлено: voice_confidence_model.pkl, scaler.pkl")


if __name__ == "__main__":
    PATH_TO_RAVDESS = "D:/archive"
    prepare_and_train(PATH_TO_RAVDESS)