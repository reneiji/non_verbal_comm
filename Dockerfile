FROM python:3.10.6-slim
WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    && pip install --upgrade pip \
    && pip install --only-binary=praat-parselmouth praat-parselmouth || pip install praat-parselmouth \
    && apt-get remove -y build-essential cmake pkg-config \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip cache purge

COPY requirements.txt .
RUN pip install -r requirements.txt && pip cache purge

COPY . /app
EXPOSE 8000
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8000"]