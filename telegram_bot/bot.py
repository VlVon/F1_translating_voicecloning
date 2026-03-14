import asyncio
import logging
import tempfile
import requests
import base64
import os
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# Configure environment variables and logging
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ORCHESTRATOR_URL = "http://orchestrator:5005/process_audio"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Please send me an audio or voice note.")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if update.message.voice:
            audio_file = update.message.voice
        elif update.message.document:
            logger.info('get.mime_type')
            mime_type = update.message.document.mime_type
            if mime_type and ('audio' in mime_type or mime_type == 'application/wav'):
                audio_file = update.message.document
                logger.info('get.audio')
        if not audio_file:
            await update.message.reply_text("Please send a valid audio/voice file.")
            return

        # Download the audio file
        file_info = await context.bot.get_file(audio_file.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            file_path = tmp.name
        await file_info.download_to_drive(file_path)
        await update.message.reply_text("Received your audio. Processing...")

        # Send to orchestrator
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(ORCHESTRATOR_URL, files={"audio": f})
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Orchestrator request failed: {str(e)}")
            await update.message.reply_text(f"Error processing audio: {str(e)}")
            return
        finally:
            os.remove(file_path)

        # Process response
        data = resp.json()
        en_list = data.get("transcripts", {}).get("english_transcript", [])
        ru_list = data.get("transcripts", {}).get("russian_transcript", [])
        audio_base64 = data.get("audio_base64", "")

        # Send transcripts
        en_text = "\n".join(en_list) if en_list else "No English transcript."
        ru_text = "\n".join(ru_list) if ru_list else "No Russian transcript."
        await update.message.reply_text(f"English Transcript:\n{en_text}")
        await update.message.reply_text(f"Russian Transcript:\n{ru_text}")

        # Handle TTS audio if present
        if audio_base64:
            try:
                mp3_data = base64.b64decode(audio_base64)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
                    mp3_tmp.write(mp3_data)
                    mp3_name = mp3_tmp.name
                await update.message.reply_audio(
                    audio=mp3_name,
                    caption="Here is the final TTS audio."
                )
                os.remove(mp3_name)
            except Exception as e:
                logger.error(f"Error processing TTS audio: {str(e)}")
                await update.message.reply_text("Error processing the TTS audio.")
        else:
            await update.message.reply_text("No final audio returned from orchestrator.")

    except Exception as e:
        logger.error(f"Unexpected error in handle_audio: {str(e)}")
        await update.message.reply_text("An unexpected error occurred while processing your audio.")


def run_bot():
    """Run the bot in a way that properly handles the event loop"""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.ALL, handle_audio))

    logger.info("Starting bot...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()