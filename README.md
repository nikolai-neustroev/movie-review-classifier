# Movie review classifier
The text sentiment classification service.
This project handles a full ML pipeline from raw data download to serving a gRPC API with DVC and TensorFlow Serving.
Try it yourself!
You may need Kaggle API token to access the data (https://www.kaggle.com/docs/api#authentication).
## Prerequisites
Make sure you have installed all the following prerequisites on your machine:

- git (tested on version 2.17.1)
- virtualenv (tested on version 15.1.0)
- dvc (tested on version 1.10.2)
- docker (tested on version 19.03.13, build 4484c46d9d)

## Getting started
### Step 0. Clone this repository
```shell
$ git clone https://github.com/nikolai-neustroev/movie-review-classifier.git
$ cd movie-review-classifier/
```
### Step 1. Create virtual environment
```shell
$ virtualenv -p python3.8 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ cp -r activate venv/bin/activate
```
### Step 2. Reproduce DVC pipeline
```shell
$ dvc repro
```
### Step 3. Run TensorFlow Serving
```shell
$ docker pull tensorflow/serving
$ docker run -p 8500:8500 \
             -p 8501:8501 \
             -v "`pwd`/models/dense:/models/movie_review" \
             -e MODEL_NAME=movie_review \
             tensorflow/serving
```
### Step 4. Run client
```shell
$ python src/clients/tf_serving_grpc_client.py
```
### Step 5. Enter a review 
0 is for negative sentiment, 1 for positive.
Enter `:q` to exit.