from flask import Flask, request, jsonify, send_file, render_template, url_for
from flask_cors import CORS
import nltk
from transformers import pipeline
import evaluate
import newspaper
from gtts import gTTS
import os

# Download necessary NLTK data
# nltk.download('punkt')

# Initialize summarization pipeline (BART model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes 

def get_article_from_url(url):
    """Fetches and extracts text from a given URL."""
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error: Could not retrieve article from URL: {e}"

def summarize_text(text):
    """Summarizes the given text using the BART model."""
    try:
        summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Error during summarization: {e}"

def evaluate_summary(original_text, summary):
    """Evaluates the summary using ROUGE metrics."""
    rouge = evaluate.load("rouge")
    try:
        results = rouge.compute(predictions=[summary], references=[original_text])
        return results
    except Exception as e:
        return {"Error": f"Error during evaluation: {e}"}

def text_to_speech(text):
    """Converts the summarized text into speech and saves it as an audio file."""
    try:
        audio_path = "static/summary.mp3"
        tts = gTTS(text=text, lang="en")
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        return f"Error during TTS conversion: {e}"

@app.route('/summarize', methods=['POST'])
def summarize_article():
    """Handles summarization requests and returns text + speech."""
    data = request.json
    input_text_or_url = data.get('input_text_or_url')

    if not input_text_or_url:
        return jsonify({"error": "No input provided"}), 400

    # Fetch and process article text if input is a URL
    if input_text_or_url.startswith("http"):
        article_text = get_article_from_url(input_text_or_url)
        if "Error" in article_text:
            return jsonify({
                "summary": None,
                "evaluation": None,
                "original_text": article_text,
                "audio_url": None
            })
    else:
        article_text = input_text_or_url

    # Generate summary, evaluate it, and create audio
    if "Error" not in article_text:
        summary = summarize_text(article_text)
        evaluation = evaluate_summary(article_text, summary)
        audio_path = text_to_speech(summary)

        return jsonify({
            "summary": summary,
            "evaluation": evaluation,
            "original_text": article_text,
            "audio_url": f"/static/summary.mp3" if "Error" not in audio_path else None
        })
    else:
        return jsonify({
            "summary": None,
            "evaluation": None,
            "original_text": article_text,
            "audio_url": None
        })

@app.route('/audio', methods=['GET'])
def get_audio():
    """Serves the generated audio file."""
    audio_path = "static/summary.mp3"
    if os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=True)
    return jsonify({"error": "Audio file not found"}), 404


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)