# Triton Custom Backend

Repository to store code and instructions for the Triton Custom Backend developments

## Build the docker container for compilation
To compile the custom backend with the triton server, we need to prepare the container first:
```
git clone git@github.com:triton-inference-server/server.git
cd server
git checkout r21.02
./build.py --build-dir=.buildir --enable-logging --enable-stats --enable-tracing --enable-metrics --enable-gpu-metrics --enable-gpu --filesystem=gcs --filesystem=azure_storage --filesystem=s3 --endpoint=http --endpoint=grpc --repo-tag=common:r21.02 --repo-tag=core:r21.02 --repo-tag=backend:r21.02 --backend=custom:r21.02 --backend=ensemble:r21.02 --backend=python:r21.02 --backend=tensorflow1:r21.02
```
This will build several containers, with the modules needed for custom, ensemble, python, and tensorflow backends. Note the pytorch, tensorRT, TF2, and onnx backends have been skipped since the focus here is on custom backend. What is needed is the final `tritonserver` container, which needs some extra libraries in order to be used to compile other custom backends:
```
nvidia-docker run -it tritonserver
apt-get update &&     apt-get install -y --no-install-recommends             autoconf             automake             build-essential             docker.io             git             libre2-dev             libssl-dev             libtool             libboost-dev             libcurl4-openssl-dev             libb64-dev             patchelf             python3-dev             python3-pip             python3-setuptools             rapidjson-dev             software-properties-common             unzip             wget             zlib1g-dev             pkg-config             uuid-dev &&     rm -rf /var/lib/apt/lists/*
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null |       gpg --dearmor - |        tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null &&     apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' &&     apt-get update &&     apt-get install -y --no-install-recommends       cmake-data=3.18.4-0kitware1ubuntu20.04.1 cmake=3.18.4-0kitware1ubuntu20.04.1
```
Then commit and save the updates for the container!

### Pre-built version
To simplify things, we have prepared the pre-built container, with the needed libraries and classes for compilation, etc here:
```
docker pull yongbinfeng/tritonserver:21.02v2
```

(The container is already at ailab01. So `docker pull` can be skipped if running on ailab01.)

## Test with Identity Custom Backend

### Compile the identity custom backend in the container

Take the identity custom backend as an example: 
```
mkdir CustomBackends
cd CustomBackends
git clone git@github.com:triton-inference-server/identity_backend.git
cd identity_backend
git checkout r21.02
```

Then start the docker container and compile inside the container:
```
nvidia-docker run -it --gpus=1 -p8020:8000 -p8021:8001 -p8022:8002 --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v/PATH_TO_CustomBackends/:/workspace/backend yongbinfeng/tritonserver:21.02v2
cd /workspace/backend/identity_backend/
mkdir build
cd build
cmake -DTRITON_ENABLE_GPU=ON -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BACKEND_REPO_TAG=r21.02 -DTRITON_CORE_REPO_TAG=r21.02 -DTRITON_COMMON_REPO_TAG=r21.02 ..
make install
exit
```

This will compile the `identity` custom backend, with the library `libtriton_identity.so`.

### Run the test with Identity backend

Clone this repository, which includes the test scripts and the model config file for `Identity`
```
mkdir Test
cd Test
git clone git@github.com:yongbinfeng/TritonCBE.git
cd TritonCBE/TestIdentity/identity_fp32/
```

#### Server side
Copy the compiled so file to the model directory:
```
cp /PATH_TO_CustomBackends/CustomBackends/identity_backend/build/libtriton_identity.so ./1/
```
Start the triton server with this `identity` model:
```
nvidia-docker run -it --gpus=1 -p8020:8000 -p8021:8001 -p8022:8002 --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v/PATH_TO_Test/TritonCBE/TestIdentity/:/models yongbinfeng/tritonserver:21.02v2 tritonserver --model-repository=/models
```
Then the server should start with the ouputs like
```
I0827 10:15:37.927419 1 grpc_server.cc:3979] Started GRPCInferenceService at 0.0.0.0:8001
I0827 10:15:37.927743 1 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I0827 10:15:37.970316 1 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```

#### Client side
Open another terminal, pull and start the client container:
```
docker pull nvcr.io/nvidia/tritonserver:21.02-py3-sdk
nvidia-docker run -it --rm -v/PATH_TO_Test/TritonCBE/TestIdentity:/workspace/test --net=host nvcr.io/nvidia/tritonserver:21.02-py3-sdk
cd /workspace/test/
python identity_test.py
python identity_test.py --protocol grpc
```

The script generates some random numpy arrays, send them to the server, and compare with the outputs from the server via https and grpc. At the end you should see:
```
Passed all tests!
```

## p2r test

### Directly running p2r
Get the [p2r](https://github.com/cerati/p2r-tests/blob/main/src/propagate-tor-test_cuda_v4.cu) code and compile inside the docker container:
```
git clone git@github.com:cerati/p2r-tests.git
nvidia-docker run -it --gpus=1 -p8020:8000 -p8021:8001 -p8022:8002 --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v/PATH_TO_p2r-test:/workspace/p2r yongbinfeng/tritonserver:21.02v2
cd /workspace/p2r
nvcc -arch=sm_70 -Iinclude -std=c++17 -maxrregcount=64 -g -lineinfo -o propagate-tor-test_cuda_v4_nvcc src/propagate-tor-test_cuda_v4.cu
```

This will produce an executble file `propagate-tor-test_cuda_v4_nvcc` in the same directory, run it with
```
./propagate-tor-test_cuda_v4_nvcc
```
The outputs should be like
```
Streams: 1, blocks: 25600, threads(x): (32), bsize = (32)
track in pos: x=-12.806847, y=-7.723825, z=38.130142, r=14.955694
track in cov: xx=6.29e-07, yy=7.53e-07, zz=9.63e-08
hit in pos: x=-20.782465, y=-12.241503, z=57.806763, r=24.119810
produce nevts=100 ntrks=8192 smearing by=0.100000
NITER=5
done preparing!
Number of struct MPTRK trk[] = 25600
Number of struct MPTRK outtrk[] = 25600
Number of struct struct MPHIT hit[] = 25600
Size of struct MPTRK trk[] = 91750400
Size of struct MPTRK outtrk[] = 91750400
Size of struct struct MPHIT hit[] = 589824000
setup time time=14.425000 (s)
done ntracks=4096000 tot time=0.414030 (s) time/trk=1.010815e-07 (s)
formatted 5 100 8192 32 256 0.414030 0 14.425000 1
track x avg=-21.287258 std/avg=0.168731
track y avg=-12.544975 std/avg=0.162441
track z avg=64.127831 std/avg=18.857132
track r avg=24.770483 std/avg=0.058340
track dx/x avg=0.012120 std=0.285085
track dy/y avg=0.013154 std=0.413136
track dz/z avg=0.083373 std=22.414892
track dr/r avg=0.015623 std=0.130872
track pt avg=3.846811
track phi avg=-2.631085
track theta avg=0.350691
```

### p2r custom backend
#### Compile the p2r custom backend
Compile the p2r custom backend inside the docker container environment:
```
nvidia-docker run -it --gpus=1 -v/PATH_TO_TritonCBE/p2rCBE:/workspace/p2r yongbinfeng/tritonserver:21.02v2
cd /workspace/p2r
mkdir build
cd build
cmake ..
make install
```
#### Test the p2r custom backend
This will produce a `libtriton_p2r.so` under build. Copy it to the directory `TritonCBE/TestP2r/p2r/1/`. Load the `p2r` model directory with the container, similar to the identity backend case:
```
nvidia-docker run -it --gpus=1 -p8030:8000 -p8031:8001 -p8032:8002 --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v/PATH_TO_TritonCBE/TestP2r:/models yongbinfeng/tritonserver:21.02v2 tritonserver --model-repository=/models
```
Start another client container, run the `p2r_test.py`. The output should be similar to
```
done preparing!
Number of struct MPTRK trk[] = 25600
Number of struct MPTRK outtrk[] = 25600
Number of struct struct MPHIT hit[] = 25600
Size of struct MPTRK trk[] = 91750400
Size of struct MPTRK outtrk[] = 91750400
Size of struct struct MPHIT hit[] = 589824000
setup time time=6.673000 (s)
done ntracks=4096000 tot time=0.475465 (s) time/trk=1.160803e-07 (s)
formatted 5 100 8192 32 256 0.475465 0 6.673000 1
track x avg=-21.287258 std/avg=0.168731
track y avg=-12.544975 std/avg=0.162441
track z avg=64.127831 std/avg=18.857132
track r avg=24.770483 std/avg=0.058340
track dx/x avg=0.012120 std=0.285085
track dy/y avg=0.013154 std=0.413136
track dz/z avg=0.083373 std=22.414892
track dr/r avg=0.015623 std=0.130872
track pt avg=3.846811
track phi avg=-2.631085
track theta avg=0.350691
```

### HCAL DigisGPU Custom Backend for MAHI
Work in progress
