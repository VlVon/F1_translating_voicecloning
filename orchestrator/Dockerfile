FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app/main.py

EXPOSE 5005
CMD ["python", "main.py"]
