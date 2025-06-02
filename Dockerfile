FROM python:3.10.6-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt


RUN pip install --upgrade pip

RUN pip install --only-binary=praat-parselmouth praat-parselmouth || \
    pip install praat-parselmouth

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8000"]
