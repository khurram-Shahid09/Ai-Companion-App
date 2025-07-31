from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Loading summarization model...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route('/upload', methods=['POST'])
def upload_audio():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    try:
        transcript = whisper_model.transcribe(path)["text"]
        summary = summarizer(transcript, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        return jsonify({
            "transcript": transcript,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

