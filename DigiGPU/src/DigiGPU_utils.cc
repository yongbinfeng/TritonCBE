#include "DigiGPU_utils.h"

namespace triton { namespace backend { namespace DigiGPU {

TRITONSERVER_Error* CheckConfig(common::TritonJson::Value& inputs, common::TritonJson::Value& outputs, const std::map<std::string, size_t>& input_map) 
{
  // There must be 10 input and 1
  // output (dummy).
  RETURN_ERROR_IF_FALSE(inputs.ArraySize() == 10, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected 10 input, got ") +
                            std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(outputs.ArraySize() == 1,
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected 1 output, got ") +
                            std::to_string(outputs.ArraySize()));

  for (int i = 0; i < 10; i++) {
    common::TritonJson::Value input;
    RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));
    // Checkout input data_type and dims
    std::string input_dtype;
    RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));

    std::vector<int64_t> input_shape;
    RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));

    std::string input_name;
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name));
    RETURN_ERROR_IF_FALSE(
        input_map.count(input_name) > 0, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("input name is not among the expected: ") + input_name);

    if (input_name == "F5_STRIDE") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected F5_STRIDE input datatype as TYPE_UINT32, got ") +
              input_dtype);

      RETURN_ERROR_IF_FALSE(input_shape[0] == 1, TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("expected input shape equal 1, got ") +
                                backend::ShapeToString(input_shape));
    }
    if (input_name == "F5_IDS") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected F5_IDS input datatype as TYPE_UINT32, got ") +
              input_dtype);
    }
    if (input_name == "F5_DATA") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_UINT16", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected F5_DATA input datatype as TYPE_UINT16, got ") +
              input_dtype);

    }
    if (input_name == "F5_NPRESAMPLES") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_INT64", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected F5_NPRESAMPLES input datatype as TYPE_UINT16, got ") +
              input_dtype);

    }
    if (input_name == "F01_STRIDE") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected F01_STRIDE input datatype as TYPE_UINT32, got ") +
              input_dtype);

      RETURN_ERROR_IF_FALSE(input_shape[0] == 1, TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("expected input shape equal 1, got ") +
                                backend::ShapeToString(input_shape));
    }
    if (input_name == "F01_IDS") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected F01_IDS input datatype as TYPE_UINT32, got ") +
              input_dtype);
    }
    if (input_name == "F01_DATA") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_UINT16", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected F01_DATA input datatype as TYPE_UINT16, got ") +
              input_dtype);

    }
    if (input_name == "F3_STRIDE") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected F3_STRIDE input datatype as TYPE_UINT32, got ") +
              input_dtype);

      RETURN_ERROR_IF_FALSE(input_shape[0] == 1, TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("expected input shape equal 1, got ") +
                                backend::ShapeToString(input_shape));
    }
    if (input_name == "F3_IDS") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected F3_IDS input datatype as TYPE_UINT32, got ") +
              input_dtype);

    }
    if (input_name == "F3_DATA") {
      RETURN_ERROR_IF_FALSE(
          input_dtype == "TYPE_UINT16", TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected F3_DATA input datatype as TYPE_UINT16, got ") +
              input_dtype);
    }
  }

  common::TritonJson::Value output;
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));
  std::string output_dtype;
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
  RETURN_ERROR_IF_FALSE(
      output_dtype == "TYPE_BOOL", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected  output datatype as TYPE_BOOL, got ") +
          output_dtype);

  //  output must have 1 shape
  std::vector<int64_t> output_shape;
  RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));
  RETURN_ERROR_IF_FALSE(output_shape[0] == 1, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected  output shape equal 1, got ") +
                            backend::ShapeToString(output_shape));

  return nullptr;  // success
}

}}}
