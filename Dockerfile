FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y \
		praat \
		libx11-dev \
		cmake \
		build-essential


WORKDIR /app

COPY . /app
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir praat-parselmouth
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["streamlit", "run", "app/app.py", "--host", "0.0.0.0"]
