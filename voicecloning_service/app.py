from flask import Flask, request, jsonify, send_file
import os
import tempfile
from pydub import AudioSegment
from elevenlabs import ElevenLabs, save
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VoiceCloningService')

ELEVENLABS_TOKEN = os.getenv("ELEVENLABS_TOKEN", "")
voicecloning_model = ElevenLabs(api_key=ELEVENLABS_TOKEN)
REFERENCE_CLONE_DIR = '/app/reference_speakers_clone'

speakers_cloning = {
    'Oscar Piastri': ['oscar_piastri_1.wav', 'oscar_piastri_2.wav'],
    'Tom Stallard': ['tom_stallard_1.wav', 'tom_stallard_2.wav'],
    'Lando Norris': ['lando_norris_1.wav', 'lando_norris_2.wav'],
    'Will Joseph': ['will_joseph_1.wav', 'will_joseph_2.wav'],
    'Max Verstappen': ['max_verstappen_1.wav', 'max_verstappen_2.wav'],
    'Gianpiero Lambiase': ['gianpiero_lambiase_1.wav', 'gianpiero_lambiase_2.wav']
}


@app.route("/clone", methods=["POST"])
def clone():
    logger.info("Received voice cloning request")
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data in request")
            return jsonify({"error": "Invalid request"}), 400

        speaker_name = data.get("speaker_name")
        text = data.get("text", "")

        logger.debug(f"Request data - Speaker: {speaker_name}, Text length: {len(text)} chars")

        if not speaker_name:
            logger.error("Missing speaker_name parameter")
            return jsonify({"error": "speaker_name is required"}), 400

        if not text:
            logger.warning("Empty text received for cloning")
            return jsonify({"error": "Text is required"}), 400

        logger.info(f"Processing clone request for {speaker_name}")
        ref_files = speakers_cloning.get(speaker_name, [])

        if not ref_files:
            logger.error(f"No reference files for {speaker_name}")
            return jsonify({"error": "Speaker not found"}), 404

        # Get full paths to reference files
        ref_paths = [os.path.join(REFERENCE_CLONE_DIR, f) for f in ref_files]
        logger.debug(f"Using reference files: {ref_paths}")

        # Generate voice clone
        try:
            logger.info("Initializing voice cloning...")
            voice = voicecloning_model.clone(
                name=f"{speaker_name}_clone",
                description="F1 radio cloned voice",
                files=ref_paths
            )

            logger.info("Generating speech...")
            audio = voicecloning_model.generate(
                text=text,
                voice=voice,
                model="eleven_multilingual_v2"
            )

            # Save temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
                save(audio, tmp_path)
                logger.debug(f"Saved temporary audio to {tmp_path}")

            logger.info("Voice cloning completed successfully")
            return send_file(
                tmp_path,
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name="cloned_voice.mp3"
            )

        except Exception as e:
            logger.error(f"Cloning process failed: {str(e)}", exc_info=True)
            return jsonify({"error": "Voice generation failed"}), 500

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug("Cleaned up temporary audio file")

    except Exception as e:
        logger.error(f"Cloning request failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting voice cloning service")
    app.run(host="0.0.0.0", port=5004)