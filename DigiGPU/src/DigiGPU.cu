/*
 code to convert HcalDigisGPU to custom backend to be run as a service
*/

#include <cuda_profiler_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <iomanip>
#include <iostream>

#include "cuda_runtime.h"

#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "triton/backend/backend_common.h"
#include "DigiGPU_utils.h"

namespace triton { namespace backend { namespace DigiGPU {

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                    \
  do {                                                                 \
    if ((RESPONSES)[IDX] != nullptr) {                                 \
      TRITONSERVER_Error* err__ = (X);                                 \
      if (err__ != nullptr) {                                          \
        LOG_IF_ERROR(TRITONBACKEND_ResponseSend(                       \
                         (RESPONSES)[IDX],                             \
                         TRITONSERVER_RESPONSE_COMPLETE_FINAL, err__), \
                     "failed to send error response");                 \
        (RESPONSES)[IDX] = nullptr;                                    \
        TRITONSERVER_ErrorDelete(err__);                               \
      }                                                                \
    }                                                                  \
  } while (false)

#define CK_CUDA_THROW_(x)                                                      \
  do {                                                                         \
    cudaError_t retval = (x);                                                  \
    if (retval != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("Runtime error: ") +                \
                               (cudaGetErrorString(retval)) + " " + __FILE__ + \
                               ":" + std::to_string(__LINE__) + " \n");        \
    }                                                                          \
  } while (0)

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model,
                                    ModelState** state);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string& Name() const { return name_; }
  uint64_t Version() const { return version_; }

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Block the thread for seconds specified in 'creation_delay_sec' parameter.
  // This function is used for testing.
  TRITONSERVER_Error* CreationDelay();

  // Get input data entry map
  std::map<std::string, size_t> GetInputmap() { return input_map_; }

 private:
  ModelState(TRITONSERVER_Server* triton_server,
             TRITONBACKEND_Model* triton_model, const char* name,
             const uint64_t version, common::TritonJson::Value&& model_config);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  const uint64_t version_;
  common::TritonJson::Value model_config_;

  bool supports_batching_;

  std::map<std::string, size_t> input_map_{
                                            {"F5_STRIDE", 0},
                                            {"F5_IDS", 1},
                                            {"F5_DATA", 2},
                                            {"F5_NPRESAMPLES", 3},
                                            {"F01_STRIDE", 4},
                                            {"F01_IDS", 5},
                                            {"F01_DATA", 6},
                                            {"F3_STRIDE", 7},
                                            {"F3_IDS", 8},
                                            {"F3_DATA", 9}
  };
};

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model,
                                       ModelState** state) {
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  common::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  TRITONSERVER_Server* triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  *state = new ModelState(triton_server, triton_model, model_name,
                          model_version, std::move(model_config));
  return nullptr;  // success
}

ModelState::ModelState(TRITONSERVER_Server* triton_server,
                       TRITONBACKEND_Model* triton_model, const char* name,
                       const uint64_t version,
                       common::TritonJson::Value&& model_config)
    : triton_server_(triton_server),
      triton_model_(triton_model),
      name_(name),
      version_(version),
      model_config_(std::move(model_config)),
      supports_batching_(false) {}

TRITONSERVER_Error* ModelState::ValidateModelConfig() {
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  CheckConfig(inputs, outputs, GetInputmap());

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);

  ~ModelInstanceState();

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance() {
    return triton_model_instance_;
  }

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Init memory 
  void InitMemory(uint32_t stride5, uint32_t stride01, uint32_t stride3);

  // execute the DigiGPU
  float ProcessRequest(TRITONBACKEND_Request* request);

  uint32_t* GetDF5IDs() {return df5_ids;}

 private:
  ModelInstanceState(ModelState* model_state,
                     TRITONBACKEND_ModelInstance* triton_model_instance,
                     const char* name,
                     const TRITONSERVER_InstanceGroupKind kind,
                     const int32_t device_id);

  ModelState* model_state_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;

  int stream_chunk_;
  int stream_remainder_;
  int stream_range_;

  cudaStream_t streams_[8];

  // device pointers
  uint16_t* df5_data;
  uint32_t* df5_ids;
  uint8_t* df5_npresamples;
  uint16_t* df01_data;
  uint32_t* df01_ids;
  uint16_t* df3_data;
  uint32_t* df3_ids;

  struct timeval timecheck_;
  long setup_start_, setup_stop_;
};

TRITONSERVER_Error* ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state) {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              std::string("ModelInstanceState::Create").c_str());
  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new ModelInstanceState(model_state, triton_model_instance,
                                  instance_name, instance_kind, instance_id);
  return nullptr;  // success
}

ModelInstanceState::~ModelInstanceState() {
  printf("calling destructor\n");
  printf("destructor done\n");
}

/*
TRITONSERVER_Error* ModelInstanceState::Init(std::vector<float>& vtrk_par, std::vector<float>& vtrk_cov, std::vector<int32_t>& vtrk_q, std::vector<float>& vhit_pos, std::vector<float>& vhit_cov) {
  ATRK inputtrk {
    {vtrk_par[0], vtrk_par[1], vtrk_par[2], vtrk_par[3], vtrk_par[4], vtrk_par[5]},
    {vtrk_cov[0], vtrk_cov[1], vtrk_cov[2], vtrk_cov[3], vtrk_cov[4], vtrk_cov[5], vtrk_cov[6], 
     vtrk_cov[7], vtrk_cov[8], vtrk_cov[9], vtrk_cov[10], vtrk_cov[11], vtrk_cov[12], vtrk_cov[13], 
     vtrk_cov[14], vtrk_cov[15], vtrk_cov[16], vtrk_cov[17], vtrk_cov[18], vtrk_cov[19], vtrk_cov[20]},
     1
  };
  AHIT inputhit = {
    {vhit_pos[0], vhit_pos[1], vhit_pos[2]},
    {vhit_cov[0], vhit_cov[1], vhit_cov[2], vhit_cov[3], vhit_cov[4], vhit_cov[5]}
  };
  {
    std::stringstream ss;
    ss << "track in pos: x=" << inputtrk.par[0] << ", y=" << inputtrk.par[1]
       << ", z=" << inputtrk.par[2] << ", r="
       << sqrtf(inputtrk.par[0] * inputtrk.par[0] +
                inputtrk.par[1] * inputtrk.par[1]);
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ss.str().c_str());
  }
  {
    std::stringstream ss;
    ss << "track in cov: xx=" << inputtrk.cov[SymOffsets66(PosInMtrx(0, 0, 6))]
       << ", yy=" << inputtrk.cov[SymOffsets66(PosInMtrx(1, 1, 6))]
       << ", zz=" << inputtrk.cov[SymOffsets66(PosInMtrx(2, 2, 6))];
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ss.str().c_str());
  }
  {
    std::stringstream ss;
    ss << "hit in pos: x=" << inputhit.pos[0] << ", y=" << inputhit.pos[1]
       << ", z=" << inputhit.pos[2] << ", r="
       << sqrtf(inputhit.pos[0] * inputhit.pos[0] +
                inputhit.pos[1] * inputhit.pos[1]);
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ss.str().c_str());
  }
  {
    std::stringstream ss;
    ss << "produce nevts=" << nevts << " ntrks=" << ntrks
       << " smearing by=" << smear;
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ss.str().c_str());
  }
  {
    std::stringstream ss;
    ss << "NITER=" << NITER;
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ss.str().c_str());
  }

  gettimeofday(&timecheck_, NULL);
  setup_start_ =
      (long)timecheck_.tv_sec * 1000 + (long)timecheck_.tv_usec / 1000;

  cudaMallocManaged((void**)&outtrk_, nevts * nb * sizeof(MPTRK));

  cudaMalloc((MPTRK**)&trk_dev_, nevts * nb * sizeof(MPTRK));
  cudaMalloc((MPHIT**)&hit_dev_, nlayer * nevts * nb * sizeof(MPHIT));
  cudaMalloc((MPTRK**)&outtrk_dev_, nevts * nb * sizeof(MPTRK));

  stream_chunk_ = ((int)(nevts * nb / num_streams));
  stream_remainder_ = ((int)((nevts * nb) % num_streams));
  if (stream_remainder_ == 0) {
    stream_range_ = num_streams;
  } else {
    stream_range_ = num_streams + 1;
  }

  for (int s = 0; s < stream_range_; s++) {
    cudaStreamCreate(&streams_[s]);
  }

  gettimeofday(&timecheck_, NULL);
  setup_stop_ =
      (long)timecheck_.tv_sec * 1000 + (long)timecheck_.tv_usec / 1000;

  printf("done preparing!\n");

  printf("Number of struct MPTRK trk[] = %d\n", nevts * nb);
  printf("Number of struct MPTRK outtrk[] = %d\n", nevts * nb);
  printf("Number of struct struct MPHIT hit[] = %d\n", nevts * nb);

  printf("Size of struct MPTRK trk[] = %ld\n",
         nevts * nb * sizeof(struct MPTRK));
  printf("Size of struct MPTRK outtrk[] = %ld\n",
         nevts * nb * sizeof(struct MPTRK));
  printf("Size of struct struct MPHIT hit[] = %ld\n",
         nlayer * nevts * nb * sizeof(struct MPHIT));

  return nullptr;
}
*/

void ModelInstanceState::InitMemory(uint32_t stride5, uint32_t stride01, uint32_t stride3) {
  CK_CUDA_THROW_(cudaMalloc((uint16_t**)&df5_data, 1e4*stride5*sizeof(uint16_t)));
  CK_CUDA_THROW_(cudaMalloc((uint32_t**)&df5_ids,  1e4*sizeof(uint32_t)));
  CK_CUDA_THROW_(cudaMalloc((uint8_t**)&df5_npresamples,  1e4*sizeof(uint8_t)));
  CK_CUDA_THROW_(cudaMalloc((uint16_t**)&df01_data, 1e4*stride01*sizeof(uint16_t)));
  CK_CUDA_THROW_(cudaMalloc((uint32_t**)&df01_ids, 1e4*sizeof(uint32_t)));
  CK_CUDA_THROW_(cudaMalloc((uint16_t**)&df3_data, 1e4*stride3*sizeof(uint16_t)));
  CK_CUDA_THROW_(cudaMalloc((uint32_t**)&df3_ids,  1e4*sizeof(uint32_t)));
}

float ModelInstanceState::ProcessRequest(TRITONBACKEND_Request* request) {
  return 0.5;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : model_state_(model_state),
      triton_model_instance_(triton_model_instance),
      name_(name),
      kind_(kind),
      device_id_(device_id) {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Triton Model Instance Initialization on device ") +
               std::to_string(device_id))
                  .c_str());
  cudaError_t cuerr = cudaSetDevice(device_id);
  if (cuerr != cudaSuccess) {
    std::cerr << "failed to set CUDA device to " << device_id << ": "
              << cudaGetErrorString(cuerr);
  }
}

extern "C" {

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInitialize: ") + name +
               " (version " + std::to_string(version) + ")")
                  .c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  const char* cname;

  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  // requires the GPU instance
  RETURN_ERROR_IF_FALSE(
      instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'DigiGPU' backend only supports GPU instances"));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("model instance ") + instance_state->Name() +
               ", executing " + std::to_string(request_count) + " requests")
                  .c_str());

  // This backend does not support models that support batching, so
  // 'request_count' should always be 1.
  RETURN_ERROR_IF_FALSE(
      request_count <= 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("repeat backend does not support batched request execution"));

  // 'responses' is initialized with the response objects below and
  // if/when an error response is sent the corresponding entry in
  // 'responses' is set to nullptr to indicate that that response has
  // already been sent.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  // Create a single response object for each request. If something
  // goes wrong when attempting to create the response objects just
  // fail all of the requests by returning an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(responses, r,
                             TRITONBACKEND_RequestId(request, &request_id));

    uint64_t correlation_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

    // Triton ensures that there is only a single input since that is
    // what is specified in the model configuration, so normally there
    // would be no reason to check it but we do here to demonstrate the
    // API.
    uint32_t input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInputCount(request, &input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    // If an error response was sent for the above then display an
    // error message and move on to next request.
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read request input/output counts, error response sent")
              .c_str());
      continue;
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("request ") + std::to_string(r) + ": id = \"" +
         request_id + "\", correlation_id = " + std::to_string(correlation_id) +
         ", input_count = " + std::to_string(input_count) +
         ", requested_output_count = " + std::to_string(requested_output_count))
            .c_str());

    const char* input_name;
    for (uint32_t i = 0; i < 10; ++i) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestInputName(request, i /* index */, &input_name));
      RETURN_ERROR_IF_FALSE(
          instance_state->StateForModel()->GetInputmap().count(input_name) > 0,
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("input name not among the expected list: ") +
              input_name);
    }

    uint32_t f5_stride;
    size_t f5_stride_size = sizeof(uint32_t);
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        backend::ReadInputTensor(request, "F5_STRIDE", reinterpret_cast<char*>(&f5_stride), &f5_stride_size));

    uint32_t f01_stride;
    size_t f01_stride_size = sizeof(uint32_t);
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        backend::ReadInputTensor(request, "F01_STRIDE", reinterpret_cast<char*>(&f01_stride), &f01_stride_size));

    uint32_t f3_stride;
    size_t f3_stride_size = sizeof(uint32_t);
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        backend::ReadInputTensor(request, "F3_STRIDE", reinterpret_cast<char*>(&f3_stride), &f3_stride_size));

    instance_state->InitMemory(f5_stride, f01_stride, f3_stride);

    TRITONBACKEND_Input* f5_ids_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, "F5_IDS", &f5_ids_input));

    TRITONBACKEND_Input* f5_data_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, "F5_DATA", &f5_data_input));

    TRITONBACKEND_Input* f5_npresamples_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, "F5_NPRESAMPLES", &f5_npresamples_input));

    TRITONBACKEND_Input* f01_ids_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, "F01_IDS", &f01_ids_input));

    TRITONBACKEND_Input* f01_data_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, "F01_DATA", &f01_data_input));

    TRITONBACKEND_Input* f3_ids_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, "F3_IDS", &f3_ids_input));

    TRITONBACKEND_Input* f3_data_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, "F3_DATA", &f3_data_input));

    // copy input data in gpu memory after init with malloc
    const void* f5_ids_buffer=nullptr;
    uint64_t buffer_byte_size = sizeof(uint32_t);
    TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_GPU;
    int64_t input_memory_type_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputBuffer(
            f5_ids_input, 0, &f5_ids_buffer, &buffer_byte_size, &input_memory_type,
            &input_memory_type_id));
    CK_CUDA_THROW_(cudaMemcpy(instance_state->GetDF5IDs(), f5_ids_buffer, buffer_byte_size, cudaMemcpyHostToDevice));

    // We also validated that the model configuration specifies only a
    // single output, but the request is not required to request any
    // output at all so we only produce an output if requested.
    const char* requested_output_name = nullptr;
    if (requested_output_count > 0) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(request, 0 /* index */,
                                          &requested_output_name));

      // prepare output
     TRITONBACKEND_Response* response = responses[r];
      TRITONBACKEND_Output* output;
      const int64_t out_putshape=1;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &output, requested_output_name, TRITONSERVER_TYPE_FP32,
              &out_putshape, 1));

      void* output_buffer;
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, sizeof(float), &output_memory_type,
              &output_memory_type_id));

      //instance_state->Init(v_trk_par, v_trk_cov, v_trk_q, v_hit_pos, v_hit_cov);
      float avgpt = instance_state->ProcessRequest(request);

      // dummy output: avgpt
      memcpy(output_buffer, &avgpt, sizeof(float));
    }

    // If we get to this point then there hasn't been any error and
    // the response is complete and we can send it. This is the last
    // (and only) response that we are sending for the request so we
    // must mark it FINAL. If there is an error when sending all we
    // can do is log it.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            nullptr /* success */),
        "failed sending response");
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    // Report statistics for the request. Note that there could
    // still be responses that have not yet been sent but those
    // cannot be captured in the statistics as they reflect only the
    // request object. We use the execution start/end time for
    // compute also so that the entire execution time is associated
    // with the inference computation.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request, true /* success */,
            exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  std::cout << "finished " << std::endl;

  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;
}

}


}  // namespace DigiGPU
}  // namespace backend
}  // namespace triton
