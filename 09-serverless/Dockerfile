FROM agrigorev/model-2025-hairstyle:v1

RUN pip install onnxruntime keras-image-helper==0.0.2

ARG MODEL_NAME=hair_classifier_empty.onnx
ENV MODEL_NAME=${MODEL_NAME}

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]