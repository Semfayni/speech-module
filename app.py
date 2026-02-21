from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from audio_analyzer import PitchAnalyzer
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ініціалізація аналізатора
analyzer = PitchAnalyzer(whisper_model_size="small")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "Файл не знайдено"}), 400

    audio_file = request.files['audio']
    topic = request.form.get('topic', 'Загальна тема')
    style = request.form.get('style', 'Презентація')

    filename = audio_file.filename or "speech.wav"
    if not filename.endswith(('.wav', '.webm', '.mp3')):
        filename += ".wav"

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        audio_file.save(file_path)
        result = analyzer.process_audio(file_path, topic, style)

        os.remove(file_path)
        return jsonify(result)

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)