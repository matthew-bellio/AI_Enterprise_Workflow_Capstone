# AI Enterprise Workflow Capstone Project

This repo contains the work done as a part of the capstone project. 

The goal was to use either supervised learning or time series to forecast revenue prediction for the next 30 days.

# Usage notes

All commands are from this directory.

## To test `app.py`

```bash
~$ python app.py
```

or to start the flask app in debug mode

```bash
~$ python app.py -d
```

Go to http://127.0.0.1:8080/ and you will see a basic website that can be customtized for a project
    
## To test the model directly and train the model

see the code at the bottom of `model.py`

```bash
~$ python model.py
```

## Run the unittests

Before running the unit tests launch the `app.py`

To run only the api tests

```bash
~$ python unittests/ApiTests.py
```

To run only the model tests

```bash
~$ python unittests/ModelTests.py
```

To run only the logger tests

```bash
~$ python unittests/LoggerTests.py
```

To run all of the tests

```bash
~$ python run-tests.py
```

## Predict using one country model via Flask app

With the Flask app running

```bash
~$ python request-one.py
```

## Predict using all countries model via Flask app

With the Flask app running

```bash
~$ python request-all.py
```

## To build the Docker container

```bash
~$ docker build -t ai_enterprise_capstone .
```

Check that the image is there.

```bash
~$ docker image ls
```

You may notice images that you no longer use. You may delete them with

```bash
~$ docker image rm IMAGE_ID_OR_NAME
```

## Run the container to test that it is working  

```bash
~$ docker run -it -p 4000:8080 ai_enterprise_capstone
```

Go to http://127.0.0.1:4000/ and you will see a basic website that can be customtized for a project

## Predict using one country model via Docker

With Docker image running

```bash
~$ python docker-request-one.py
```

## Predict using all countries model via Docker

With Docker image running

```bash
~$ python docker-request-all.py
```

## Notebooks

The `AAVAIL_EDA` notebook contains the code for exploratory analysis and data visualization

The `AAVAIL_Modeling` notebook contains the code for testing multiple pipelines and models

The `Performance` notebook contains the code for monitoring the performance of the selected model