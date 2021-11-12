Step 1: Setup all the baseline code for this. You will need patatrack standalone, the Triton back end, and the server config file

```
export BASEDIR=`pwd` 
git clone -b v21.02_phil https://github.com/violatingcp/identity_backend.git 
git clone -b 21.02_phil https://github.com/violatingcp/pixeltrack-standalone.git 
git clone https://github.com/violatingcp/TritonCBE.git 
```
There are in fact three different branches of configurations for the pixel track standlone, and three sets of configurations for the identity backend. The above is the default, which enables the highest throughput with CMSSW. However if you would like to run performance client tests with the same default setup. Please checkout. 
```
git clone -b v21.02_phil_standalone https://github.com/violatingcp/identity_backend.git 
git clone -b 21.02_phil_standalone https://github.com/violatingcp/pixeltrack-standalone.git 
```
Finally, if you are interested in using the dynamic batching impelmentation. 
```
git clone -b v21.02_phil_standalone_dynamic https://github.com/violatingcp/identity_backend.git 
git clone -b 21.02_phil_standalone_dynamic https://github.com/violatingcp/pixeltrack-standalone.git 
git clone https://github.com/violatingcp/TritonCBE.git 
```
Once you have done the above, the following instructions apply to all branches. The specific perf and standalone tests will be listed below. 


nvidia-docker run -it --gpus=1 -p8020:8000 -p8021:8001 -p8022:8002 --rm -v$BASEDIR/pixeltrack-standalone/:/workspace/backend/pixel/ yongbinfeng/tritonserver:21.02v2 \
cd /workspace/backend/pixel/ \
make -j`nproc` cudadev 

nvidia-docker run -it --gpus=1 -p8020:8000 -p8021:8001 -p8022:8002 --rm -v$BASEDIR/identity_backend/:/workspace/backend/ yongbinfeng/tritonserver:21.02v2 \ 
cd /workspace/backend/ \
mkdir build \
cd build \
cmake -DTRITON_ENABLE_GPU=ON -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BACKEND_REPO_TAG=r21.02 -DTRITON_CORE_REPO_TAG=r21.02 -DTRITON_COMMON_REPO_TAG=r21.02 .. \
make install 

cp $BASEDIR/pixeltrack-standalone/lib/cudadev/*.so $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/ \
cp $BASEDIR/identity_backend/build/libtriton_identity.so                     $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/ \
cp $BASEDIR/pixeltrack-standalone/external/tbb/lib/libtbb.so*                $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/ \
cp $BASEDIR/pixeltrack-standalone/external/libbacktrace/lib/libbacktrace.so  $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/ \
cd $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/
wget data.tgz https://www.dropbox.com/s/c9pzz0k0h8ng5wd/data.tgz?dl=0  \
mv data.tgz?dl=0  data.tgz \
tar xzvf data.tgz \
 
nvidia-docker run -it --gpus=1 -p8020:8000 -p8021:8001 -p8022:8002 --rm -v$BASEDIR/TritonCBE/TestIdentity/:/models yongbinfeng/tritonserver:21.02v2 \
export LD_LIBRARY_PATH="/models/identity_fp32/1/:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
export LD_PRELOAD="/models/identity_fp32/1/libFramework.so:/models/identity_fp32/1/libCUDACore.so:/models/identity_fp32/1/libtbb.so.2:/models/identity_fp32/1/libCUDADataFormats.so:/models/identity_fp32/1/libCondFormats.so:/models/identity_fp32/1/pluginBeamSpotProducer.so:/models/identity_fp32/1/pluginSiPixelClusterizer.so:/models/identity_fp32/1/pluginValidation.so:/models/identity_fp32/1/pluginPixelTriplets.so:/models/identity_fp32/1/pluginPixelTrackFitting.so::/models/identity_fp32/1/pluginPixelVertexFinding.so:pluginSiPixelRecHits.so:/models/identity_fp32/1/libCUDADataFormats.so" \

tritonserver --model-repository=/models/ \
