# F1 Radio Translator

End-to-end Formula 1 radio communications pipeline with diarization, transcription, translation, and voice cloning delivered through a Telegram-based workflow.

## Overview

This project is designed to process Formula 1 team radio communications in a fully structured pipeline.  
It separates speakers, transcribes audio, translates the content, and generates voice-cloned output through a microservice architecture.

The system is intended for automated handling of F1-style radio messages between drivers and race engineers.

## Main Features

- Speaker diarization
- Speech-to-text transcription
- Translation pipeline
- Voice cloning
- Telegram-based interaction
- Microservice architecture
- Docker Compose orchestration

## Project Structure

F1_translating_voicecloning/
├── orchestrator/
├── transcription/
├── translation/
├── voice_cloning/
├── diarization/
├── README.md
├── docker-compose.yml
└── .gitignore
