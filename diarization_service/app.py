from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import torchaudio
import torch
from pyannote.audio import Pipeline, Audio
from speechbrain.inference.speaker import EncoderClassifier
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DiarizationService')

HF_TOKEN = os.getenv('HF_TOKEN', '')
reference_speakers_dir = '/app/reference_speakers_emb'

# Initialize models
try:
    logger.info("Initializing diarization model...")
    diarization_model = Pipeline.from_pretrained(
        'pyannote/speaker-diarization-3.1',
        use_auth_token=HF_TOKEN
    )
    logger.info("Diarization model loaded successfully")

    logger.info("Initializing speaker identification model...")
    identification_model = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb'
    )
    logger.info("Speaker identification model loaded successfully")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise

speakers = {
    'Oscar Piastri': 'oscar_piastri_emb.wav',
    'Tom Stallard': 'tom_stallard_emb.wav',
    'Lando Norris': 'lando_norris_emb.wav',
    'Will Joseph': 'will_joseph_emb.wav',
    'Max Verstappen': 'max_verstappen_emb.wav',
    'Gianpiero Lambiase': 'gianpiero_lambiase_emb.wav'
}
speaker_embeddings = {}


def compute_embedding(filepath):
    try:
        waveform, sr = torchaudio.load(filepath, normalize=True)
        emb = identification_model.encode_batch(waveform).flatten()
        return emb
    except Exception as e:
        logger.error(f"Embedding computation failed for {filepath}: {str(e)}")
        raise


def load_embeddings():
    logger.info("Loading speaker embeddings...")
    for name, embfile in speakers.items():
        path = os.path.join(reference_speakers_dir, embfile)
        if os.path.exists(path):
            try:
                speaker_embeddings[name] = compute_embedding(path)
                logger.debug(f"Successfully loaded embedding for {name}")
            except Exception as e:
                logger.error(f"Failed to load embedding for {name}: {str(e)}")
                speaker_embeddings[name] = None
        else:
            logger.warning(f"Embedding file not found for {name}: {path}")
            speaker_embeddings[name] = None
    logger.info("Completed loading speaker embeddings")


load_embeddings()


@app.route("/diarize", methods=["POST"])
def diarize_audio():
    logger.info("Received diarization request")
    try:
        if 'audio' not in request.files:
            logger.error("No audio file found in request")
            return jsonify({"error": "No 'audio' provided"}), 400

        file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            file.save(tmp.name)
            input_path = tmp.name
        logger.debug(f"Saved temporary audio file: {input_path}")

        audio_loader = Audio(mono=True, sample_rate=16000)
        waveform, sr = audio_loader(input_path)

        logger.info("Performing diarization...")
        diar_result = diarization_model(
            {"waveform": waveform, "sample_rate": sr},
            num_speakers=2
        )
        logger.info(f"Diarization completed with {len(diar_result)} segments")

        segments = []
        for turn, _, spk_label in diar_result.itertracks(yield_label=True):
            start_t = turn.start
            end_t = turn.end

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg_file:
                seg_path = seg_file.name

            slice_cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ss", str(start_t),
                "-to", str(end_t),
                "-c", "copy",
                seg_path
            ]

            logger.debug(f"Extracting segment {start_t}-{end_t}")
            subprocess.run(
                slice_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            try:
                speaker_name, score = identify_speaker(seg_path)
                logger.debug(f"Identified speaker: {speaker_name} (score: {score:.2f})")
            except Exception as e:
                logger.error(f"Speaker identification failed: {str(e)}")
                speaker_name = "UNIDENTIFIED"
                score = 0.0

            os.remove(seg_path)

            segments.append({
                "start": start_t,
                "end": end_t,
                "diarization_label": spk_label,
                "identified_speaker": speaker_name,
                "confidence_score": score
            })

        os.remove(input_path)
        logger.info(f"Returning {len(segments)} diarization segments")
        return jsonify({"segments": segments})

    except Exception as e:
        logger.error(f"Diarization failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


def identify_speaker(path_segment):
    try:
        seg_emb = compute_embedding(path_segment)
        best_speaker = "UNIDENTIFIED"
        best_score = -999.0
        threshold = 0.2

        for spk, ref_emb in speaker_embeddings.items():
            if ref_emb is None:
                continue
            score = torch.nn.functional.cosine_similarity(
                seg_emb,
                ref_emb,
                dim=0
            ).item()

            if score > best_score:
                best_score = score
                best_speaker = spk

        if best_score < threshold:
            logger.debug(f"No speaker met threshold ({best_score:.2f} < {threshold})")
            best_speaker = "UNIDENTIFIED"

        return best_speaker, best_score
    except Exception as e:
        logger.error(f"Speaker identification failed: {str(e)}")
        return "UNIDENTIFIED", 0.0


if __name__ == "__main__":
    logger.info("Starting diarization service")
    app.run(host="0.0.0.0", port=5001)