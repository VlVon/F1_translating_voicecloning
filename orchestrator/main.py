from flask import Flask, request, jsonify
import requests
import os
import tempfile
import subprocess
from pydub import AudioSegment
from collections import Counter
import base64
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AudioProcessingService')

# Microservice endpoints
DIAR_URL = "http://diarization_service:5001/diarize"
STT_URL = "http://transcription_service:5002/transcribe"
TRANS_URL = "http://translation_service:5003/translate"
CLONE_URL = "http://voicecloning_service:5004/clone"

# Speaker pairs configuration
pairs = {
    'Oscar Piastri': 'Tom Stallard',
    'Tom Stallard': 'Oscar Piastri',
    'Lando Norris': 'Will Joseph',
    'Will Joseph': 'Lando Norris',
    'Max Verstappen': 'Gianpiero Lambiase',
    'Gianpiero Lambiase': 'Max Verstappen'
}


@app.route("/process_audio", methods=["POST"])
def process_audio():
    logger.info("Starting audio processing request")
    input_path = None
    try:
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({"error": "No 'audio' file provided"}), 400

        # Save input file
        file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            file.save(tmp.name)
            input_path = tmp.name
        logger.info(f"Saved input audio to {input_path}")

        # Step 1: Diarization
        logger.info("Starting diarization...")
        with open(input_path, "rb") as f:
            diar_resp = requests.post(DIAR_URL, files={"audio": f})

        if diar_resp.status_code != 200:
            logger.error(f"Diarization failed: {diar_resp.text}")
            return jsonify({
                "error": "Diarization service error",
                "details": diar_resp.text
            }), 500

        diar_data = diar_resp.json()
        segments = diar_data.get("segments", [])
        logger.info(f"Received {len(segments)} diarization segments")

        # Step 1.5: Speaker pair correction
        logger.info("Applying speaker pair correction")
        recognized_speakers = [
            seg["identified_speaker"]
            for seg in segments
            if seg["identified_speaker"] != "UNIDENTIFIED"
        ]
        counts = Counter(recognized_speakers)

        if counts:
            majority_speaker, majority_count = counts.most_common(1)[0]
            pair_speaker = pairs.get(majority_speaker, "UNIDENTIFIED")
            logger.info(f"Majority speaker: {majority_speaker} (count: {majority_count}), pair: {pair_speaker}")

            modified_count = 0
            for seg in segments:
                current_speaker = seg["identified_speaker"]
                if current_speaker not in [majority_speaker, pair_speaker]:
                    seg["identified_speaker"] = pair_speaker
                    modified_count += 1
            logger.info(f"Modified {modified_count} speaker labels")

        # Initialize processing results
        en_transcripts = []
        ru_transcripts = []
        combined_tts = AudioSegment.empty()
        processed_segments = 0

        # Step 2: Process each segment
        logger.info(f"Processing {len(segments)} audio segments")
        for idx, seg in enumerate(segments):
            try:
                logger.info(f"Processing segment {idx + 1}/{len(segments)}")
                start = seg["start"]
                end = seg["end"]
                speaker_name = seg["identified_speaker"]

                # Extract audio segment
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg_file:
                    seg_path = seg_file.name

                ff_cmd = [
                    "ffmpeg", "-y",
                    "-i", input_path,
                    "-ss", str(start),
                    "-to", str(end),
                    "-c", "copy",
                    seg_path
                ]
                logger.debug(f"Executing: {' '.join(ff_cmd)}")
                subprocess.run(
                    ff_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                # Speech-to-Text
                logger.debug(f"Segment {idx + 1}: Starting STT")
                with open(seg_path, "rb") as audio_seg:
                    stt_resp = requests.post(STT_URL, data=audio_seg.read())

                if stt_resp.status_code != 200:
                    logger.warning(f"STT failed for segment {idx + 1}")
                    text_en = "TRANSCRIPTION_FAILED"
                else:
                    stt_data = stt_resp.json()
                    text_en = stt_data.get("text", "")
                en_transcripts.append(f"{speaker_name}: {text_en}")

                # Translation
                logger.debug(f"Segment {idx + 1}: Starting translation")
                trans_resp = requests.post(
                    TRANS_URL,
                    json={"text": text_en, "target_lang": "RU"}
                )

                if trans_resp.status_code != 200:
                    logger.warning(f"Translation failed for segment {idx + 1}")
                    text_ru = "TRANSLATION_FAILED"
                else:
                    trans_data = trans_resp.json()
                    text_ru = trans_data.get("translated_text", "")
                ru_transcripts.append(f"{speaker_name} (RU): {text_ru}")

                # Voice Cloning
                logger.debug(f"Segment {idx + 1}: Starting voice cloning")
                clone_resp = requests.post(
                    CLONE_URL,
                    json={
                        "speaker_name": speaker_name,
                        "text": text_ru
                    }
                )

                if clone_resp.status_code == 200:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tts_temp:
                        tts_temp.write(clone_resp.content)
                        tts_temp.flush()

                    tts_segment = AudioSegment.from_mp3(tts_temp.name)
                    combined_tts += tts_segment
                    os.remove(tts_temp.name)
                    logger.debug(f"Segment {idx + 1}: TTS added successfully")
                else:
                    logger.warning(f"Cloning failed for segment {idx + 1}")

                os.remove(seg_path)
                processed_segments += 1

            except Exception as seg_error:
                logger.error(f"Error processing segment {idx + 1}: {str(seg_error)}")
                continue

        # Final processing
        logger.info("Generating final output")
        final_mp3_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        combined_tts.export(final_mp3_file.name, format="mp3")
        logger.debug(f"Exported combined audio to {final_mp3_file.name}")

        with open(final_mp3_file.name, "rb") as f:
            mp3_data = f.read()
        os.remove(final_mp3_file.name)

        audio_base64 = base64.b64encode(mp3_data).decode("utf-8")
        logger.info(f"Processed {processed_segments}/{len(segments)} segments successfully")

        return jsonify({
            "transcripts": {
                "english_transcript": en_transcripts,
                "russian_transcript": ru_transcripts
            },
            "audio_base64": audio_base64
        })

    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal processing error"}), 500

    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
            logger.debug("Cleaned up input file")


if __name__ == "__main__":
    logger.info("Starting audio processing service")
    app.run(host="0.0.0.0", port=5005)