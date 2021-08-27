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

(The container is already at ailab01. So `docker pull` can be skipped if running on ailab01.)

## Compile the custom backend in the container

Take the identity custom backend as an example: 
```
mkdir CustomBackends
git clone git@github.com:triton-inference-server/identity_backend.git
cd identity_backend
git checkout r21.02
```

Then start the docker container and compile inside the container:
```
nvidia-docker run -it --gpus=1 -p8020:8000 -p8021:8001 -p8022:8002 --rm -v/PATH_TO_CustomBackends/:/workspace/backend yongbinfeng/tritonserver:21.02v2
cd /workspace/backend/identity_backend/
mkdir build
cd build
cmake -DTRITON_ENABLE_GPU=ON -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BACKEND_REPO_TAG=r21.02 -DTRITON_CORE_REPO_TAG=r21.02 -DTRITON_COMMON_REPO_TAG=r21.02 ..
make install
exit
```

This will compile the `identity` custom backend, with the library `libtriton_identity.so`.

## Run the test with Identity backend

Clone this repository, which includes the test scripts and the model config file for `Identity`
```
mkdir Test
git clone git@github.com:yongbinfeng/TritonCBE.git
cd TritonCBE/TestIdentity/identity_fp32/
```

### Server side
Copy the compiled so file to the model directory:
```
cp /PATH_TO_CustomBackends/CustomBackends/identity_backend/build/libtriton_identity.so ./1/
```
Start the triton server with this `identity` model:
```
nvidia-docker run -it --gpus=1 -p8020:8000 -p8021:8001 -p8022:8002 --rm -v/PATH_TO_Test/TritonCBE/TestIdentity/:/models yongbinfeng/tritonserver:21.02v2 tritonserver --model-repository=/models
```
Then the server should start with the ouputs like
```
I0827 10:15:37.927419 1 grpc_server.cc:3979] Started GRPCInferenceService at 0.0.0.0:8001
I0827 10:15:37.927743 1 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I0827 10:15:37.970316 1 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```

### Client side
Open another terminal, pull and start the client container:
```
docker pull nvcr.io/nvidia/tritonserver:21.02-py3-sdk
nvidia-docker run -it --rm -v/PATH_TO_Test/TritonCBE/TestIdentity:/workspace/test --net=host nvcr.io/nvidia/tritonserver:21.02-py3-sdk
cd /workspace/test/
python identity_test.py
python identity_test.py --protocol grpc
```

The script generates some random numpy arrays, send them to the server, and compare with the outputs from the server. At the end you should see
```
Passed all tests!
```

