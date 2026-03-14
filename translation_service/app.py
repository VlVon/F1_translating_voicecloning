from flask import Flask, request, jsonify
import os
import deepl
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TranslationService')

DEEPL_TOKEN = os.getenv("DEEPL_TOKEN", "")
translator = deepl.Translator(DEEPL_TOKEN)


@app.route("/translate", methods=["POST"])
def translate_text():
    logger.info("Received translation request")
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data in request")
            return jsonify({"error": "No data provided"}), 400

        text = data.get("text", "")
        target_lang = data.get("target_lang", "RU").upper()
        logger.debug(f"Translating text: {text[:50]}... to {target_lang}")

        if not text:
            logger.warning("Empty text received for translation")
            return jsonify({"translated_text": ""})

        logger.info("Performing translation...")
        result = translator.translate_text(
            text,
            target_lang=target_lang
        )

        translated_text = result.text
        logger.info(f"Translation successful: {translated_text[:50]}...")
        return jsonify({"translated_text": translated_text})

    except deepl.DeepLException as e:
        logger.error(f"DeepL API error: {str(e)}")
        return jsonify({"error": "Translation service error"}), 500
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting translation service")
    app.run(host="0.0.0.0", port=5003)