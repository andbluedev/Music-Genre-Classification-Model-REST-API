FROM tensorflow/tensorflow:2.2.2-py3

WORKDIR /api

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]