FROM python:3.10.6-buster

WORKDIR /app

COPY . /app
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

EXPOSE 8000

CMD ["python", "streamlit run app/app.py", "--host 0.0.0.0"]
