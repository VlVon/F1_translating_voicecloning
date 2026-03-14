from flask import Flask, request, jsonify
import requests
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TranscriptionService')

HF_TOKEN = os.getenv("HF_TOKEN", "")
API_STT_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.route("/transcribe", methods=["POST"])
def transcribe():
    logger.info("Received transcription request")
    try:
        audio_bytes = request.data
        logger.debug(f"Received audio data: {len(audio_bytes)} bytes")

        logger.info("Sending to Whisper API...")
        response = requests.post(API_STT_URL, headers=HEADERS, data=audio_bytes)
        logger.debug(f"API response status: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Transcription failed: {response.text}")
            return jsonify({"error": "Transcription API error"}), 500

        result = response.json()
        logger.debug(f"Raw API response: {result}")

        # Handle different response formats
        if isinstance(result, dict):
            text = result.get('text', '')
        elif isinstance(result, list) and len(result) > 0:
            text = result[0].get('text', '')
        else:
            text = ''
            logger.warning("Unexpected API response format")

        logger.info(f"Transcription successful: {text[:50]}...")
        return jsonify({"text": text})

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during transcription: {str(e)}")
        return jsonify({"error": "Network error"}), 500
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting transcription service")
    app.run(host="0.0.0.0", port=5002)