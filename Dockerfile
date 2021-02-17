FROM tensorflow/tensorflow:2.2.2-py3

WORKDIR /api

COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:savoury1/graphics && \
    add-apt-repository ppa:savoury1/multimedia && \
    add-apt-repository ppa:savoury1/ffmpeg4 && \
    apt-get update && apt-get upgrade -y && apt-get install -y \
        gcc \
        libc-dev \
	    cmake \
        libxslt-dev \
        libxml2-dev \
        libffi-dev \
        libssl-dev \
	        libsndfile-dev \
	    ffmpeg \
	&& rm -rf /var/lib/apt/lists/*                                 

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]