#pragma once

#include "triton/core/tritonserver.h"
#include "triton/backend/backend_common.h"

#include <map>

namespace triton { namespace backend { namespace DigiGPU {

TRITONSERVER_Error* CheckConfig(common::TritonJson::Value& inputs, common::TritonJson::Value& outputs, const std::map<std::string, size_t>& input_map);


}}}
