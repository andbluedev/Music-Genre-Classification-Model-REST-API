# Music-Genre-Classification-Model-REST-API

## Context

This project aims to predict the genre of a given song file using Data Science and Machine Learning techniques.

This repository contains the REST API built using Fast API that serves the [Tensorflow](https://github.com/tensorflow/tensorflow) model and [librosa](https://github.com/librosa/librosa) to extract features from uploaded MP3 files.

## Live Demo of the best Model

![https://res.cloudinary.com/djeszd2cw/image/upload/v1613837192/classify/classify-screen_egx3at.png](https://res.cloudinary.com/djeszd2cw/image/upload/v1613837192/classify/classify-screen_egx3at.png)

- Web application: [https://classify.k8s.pouretadev.com/](https://classify.k8s.pouretadev.com/)

## Related Repositories

- Frontend - Web App: [https://github.com/andbluedev/Music-Genre-Classification-Model-Web-APP ](https://github.com/andbluedev/Music-Genre-Classification-Model-Web-APP)
- Project Notebooks: [https://github.com/andbluedev/Music-Genre-Classification-Notebooks](https://github.com/andbluedev/Music-Genre-Classification-Notebooks)

## Development

It is recommended to use a virtual python environment in order to install every dependency.

Create a virtual environment the _venv_ directory at the root of this project.

### Installing dependencies

``` 

python3 -m venv venv
```

To activate the created virtual environment

``` 

source venv/bin/activate
```

Installing dependencies:

``` 

pip install -r requirements.txt
```

### Running the APP

``` 
uvicorn app.main:app --reload --port 5000
```
*note*: any other availabel port can be used

## Production

The Fast API for production uses Docker, python and uvicorn and is deployed on a kubernetes cluster.

### Locally build Docker Image

``` 
docker build -t music-genre-prediction-model-rest-api .
```

### Run Docker image

```
docker run -p 5000:5000  music-genre-prediction-model-rest-api
```
*note*: Add `-d`  to this command if you wan't to run the app in detached mode (running in the background leaving the current shell available for other commands).

## Documentation

The API documentation can be found [here](localhost:5000/docs) while the app is running locally (assuming it is running on localhost:5000).

