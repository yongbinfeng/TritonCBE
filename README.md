# Triton Custom Backend

Repository to store code and instructions for the Triton Custom Backend developments

## Build the container with necessary libraries for the server and the custom backend

To compile the custom backend with the triton server, we need to prepare the container first:
```
git clone git@github.com:triton-inference-server/server.git
git checkout r21.02
./build.py --build-dir=/storage/local/data1/home/yfeng/Backend/buildir --enable-logging --enable-stats --enable-tracing --enable-metrics --enable-gpu-metrics --enable-gpu --filesystem=gcs --filesystem=azure_storage --filesystem=s3 --endpoint=http --endpoint=grpc --repo-tag=common:r21.02 --repo-tag=core:r21.02 --repo-tag=backend:r21.02 --backend=custom:r21.02 --backend=ensemble:r21.02 --backend=python:r21.02 --backend=tensorflow1:r21.02 --backend=identity:r21.02
```

This will build several containers, together with the identity custom backend. Note the pytorch, tensorRT, TF2, and onnx backends have been skipped since the focus here is on custom backend.

I have prepared the container, with the needed libraries and classes for compilation, etc here:
```
docker pull yongbinfeng/tritonserver:21.02v2
```
