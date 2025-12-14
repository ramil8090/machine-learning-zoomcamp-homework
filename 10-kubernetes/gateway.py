#!/usr/bin/env python
# coding: utf-8

import os
import grpc

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from keras_image_helper import create_preprocessor

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from proto import np_to_protobuf

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(299, 299))


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'clothing-model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_layer_6'].CopyFrom(np_to_protobuf(X))
    return pb_request


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

def prepare_response(pb_response):
    preds = pb_response.outputs['output_0'].float_val
    return dict(zip(classes, preds))


def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response


app = FastAPI(title="gateway")

class PredictRequest(BaseModel):
    url: str

@app.post('/predict')
def predict_endpoint(req: PredictRequest):
    return predict(req.url)

if __name__ == '__main__':
    # url = 'http://bit.ly/mlbookcamp-pants'
    # response = predict(url)
    # print(response)
    uvicorn.run(app, host='0.0.0.0', port=9696)