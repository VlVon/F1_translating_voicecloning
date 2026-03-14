FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

EXPOSE 5004
CMD ["python", "app.py"]
