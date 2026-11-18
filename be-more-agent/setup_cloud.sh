#!/bin/bash
set -e
sudo apt update
sudo apt install -y python3-venv python3-tk portaudio19-dev libasound2-dev alsa-utils mpg123 libopenblas-dev
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "Done. Edit .env and add your free GEMINI_API_KEY and GROQ_API_KEY, then run: source venv/bin/activate && python agent.py"
