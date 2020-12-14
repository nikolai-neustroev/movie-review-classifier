# A slightly modified version of this script:
# https://github.com/snehankekre/Deploy-Deep-Learning-Models-TF-Serving-Docker/blob/master/tf_serving_grpc_client.py

import sys
import grpc
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2, get_model_metadata_pb2


def get_stub(host='127.0.0.1', port='8500'):
    channel = grpc.insecure_channel('127.0.0.1:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub


def get_model_prediction(model_input, stub, model_name):
    """ no error handling at all, just poc"""
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(model_input))
    response = stub.Predict.future(request, 5.0)  # 5 seconds
    return response.result().outputs["dense_2"].float_val


def get_model_version(model_name, stub):
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name = model_name
    request.metadata_field.append("signature_def")
    response = stub.GetModelMetadata(request, 10)
    return response.model_spec.version.value


if __name__ == '__main__':
    print("\nCreate RPC connection ...")
    stub = get_stub()
    while True:
        print("\nEnter a movie review [:q for Quit]")
        if sys.version_info[0] <= 3:
            sentence = raw_input() if sys.version_info[0] < 3 else input()
        if sentence == ':q':
            break
        model_input = [[sentence]]
        model_prediction = get_model_prediction(model_input, stub, model_name='movie_review')
        print("The model predicted ...")
        print(model_prediction)
