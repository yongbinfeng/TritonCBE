## Instructions to run Patatrack-AAS at Purdue

### Basic setup 
Purdue nodes has some pre-set environments and one needs to clean these and set up the CMSSW-related environments beforehand. To do this,
source [this script](https://github.com/yongbinfeng/TritonCBE/blob/main/TestIdentity/setup.sh):

```
source setup.sh
```

The disk space under `/home/` is limited and usually the large files can be placed under:
```
/depot/cms/users/$USER
```

Purdue nodes does not have Docker support and one can only run Singularity. So pull the needed docker containers first and run with Singularity later:

```
cd /depot/cms/users/$USER
singularity pull triton_cbe_21.04.sif docker://yongbinfeng/tritonserver:21.04
singularity pull triton_21.04.sif docker://nvcr.io/nvidia/tritonserver:21.04-py3
```

### Compile and run PatatrackAAS

Setup all the baseline code for this. You will need patatrack standalone, the Triton back end, and the server config file. (Better to run these under `/depot/cms/users/$USER` in case it runs out of space.)

```
export BASEDIR=`pwd` 
git clone -b 21.02_phil_asynch_12_3_X_port git@github.com:yongbinfeng/identity_backend.git
git clone -b 21.02_phil_asynch_12_3_X_port git@github.com:yongbinfeng/pixeltrack-standalone.git
git clone git@github.com:yongbinfeng/TritonCBE.git
```

Once you have done the above, the following instructions apply to all branches. The specific perf and standalone tests will be listed below. To compile everything
First compile the standalone patatrack. As a note this standalone has been fully synched with CMSSW_12_3_0_pre4.

```
singularity run --nv -e --no-home -B $BASEDIR/pixeltrack-standalone/:/workspace/backend/pixel /depot/cms/users/$USER/triton_cbe_21.04.sif
cd /workspace/backend/pixel/ 
make -j`nproc` cudadev 
```

Now compile the backend

```
singularity run --nv -e --no-home -B $BASEDIR/identity_backend/:/workspace/backend/ /depot/cms/users/$USER/triton_cbe_21.04.sif
cd /workspace/backend/ 
mkdir build 
cd build 
cmake -DTRITON_ENABLE_GPU=ON -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BACKEND_REPO_TAG=r21.02 -DTRITON_CORE_REPO_TAG=r21.02 -DTRITON_COMMON_REPO_TAG=r21.02 .. 
make install 
```

Now lets build the server setup. We point all the shared object files into the server directory. This is hacky, but it works. 

```
cp $BASEDIR/pixeltrack-standalone/lib/cudadev/*.so $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/ 
cp $BASEDIR/identity_backend/build/libtriton_identity.so                     $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/ 
cp $BASEDIR/pixeltrack-standalone/external/tbb/lib/libtbb.so*                $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/ 
cp $BASEDIR/pixeltrack-standalone/external/libbacktrace/lib/libbacktrace.so  $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/ 
cd $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/
wget data.tgz https://www.dropbox.com/s/o91gcntmnizh54p/data.tar.gz?dl=0
mv data.tgz?dl=0  data.tgz 
tar xzvf data.tgz 
cp $BASEDIR/pixeltrack-standalone/data/beamspot.bin data/
```

Finally, we are now ready to launch the server. The key issue here is you need to point to the shared object libraries with the LD_Preload path seen belwo. 

```
singularity run --nv -e --no-home -B $BASEDIR/TritonCBE/TestIdentity/:/models /depot/cms/users/$USER/triton_21.04.sif
export LD_LIBRARY_PATH="/models/identity_fp32/1/":$LD_LIBRARY_PATH
export LD_PRELOAD="/models/identity_fp32/1/libFramework.so:/models/identity_fp32/1/libCUDACore.so:/models/identity_fp32/1/libtbb.so.2:/models/identity_fp32/1/libCUDADataFormats.so:/models/identity_fp32/1/libCondFormats.so:/models/identity_fp32/1/pluginBeamSpotProducer.so:/models/identity_fp32/1/pluginSiPixelClusterizer.so:/models/identity_fp32/1/pluginValidation.so:/models/identity_fp32/1/pluginPixelTriplets.so:/models/identity_fp32/1/pluginPixelTrackFitting.so::/models/identity_fp32/1/pluginPixelVertexFinding.so:pluginSiPixelRecHits.so:/models/identity_fp32/1/libCUDADataFormats.so" 
tritonserver --backend-config=tensorflow,version=2 --model-repository=/models
```
That will get the server running, but there are a few things that you might want to do to check the performance. 

If you have compiled the standalone projects above, you can use the standalone Patatrack to get the local throughput to do that. To do this, change [this](https://github.com/yongbinfeng/pixeltrack-standalone/blob/21.02_phil_asynch_12_3_X_port/src/cudadev/plugin-BeamSpotProducer/BeamSpotToCUDA.cc#L31) line to
```
  :   \\beamSpotPODToken_{reg.consumes<BeamSpotPOD>()},
```

and the following `bsHost` to
```
  *bsHost = iSetup.get<BeamSpotPOD>();
  //*bsHost = iEvent.get(beamSpotPODToken_);
```
And then recompile; run:
```
cp $BASEDIR/TritonCBE/TestIdentity/identity_fp32/1/data/*.bin $BASEDIR/pixeltrack-standalone/data/
singularity run --nv -e --no-home -B $BASEDIR/pixeltrack-standalone/:/workspace/backend/pixel/ /depot/cms/users/$USER/triton_cbe.sif
cd /workspace/backend/pixel/ 
source env.sh
export LD_LIBRARY_PATH="/workspace/backend/pixel/lib/cudadev/":$LD_LIBRARY_PATH
./cudadev  #This is barebones
./cudadev --numberOfThreads 10 # 10 threads
./cudadev --numberOfThreads 10 --transfer # including transfer off GPU
./cudadev --numberOfThreads 10 --transfer --validation # including packaging of the output with track cleaning
```

Finally, here is the recipe to install the cmssw client. It is largely based on the documentation here https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGlobalHLT
```
cmsrel CMSSW_12_3_0_pre4
cd CMSSW_12_3_0_pre4/src
cmsenv
git cms-init
git cms-merge-topic yongbinfeng:PatatrackAAS_12_3_0_pre4
scramv1 b -j 8
cd test
```
Change the data files to the corresponding ones in the cluster in the script `hlt_customised.py`. On lxplus the files can be found under
```
/eos/cms/store/user/yofeng/Run2018D/
```

Now you should have everything you need to run Patatrack and PatatatrackAAS




