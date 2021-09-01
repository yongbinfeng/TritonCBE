cmake_minimum_required (VERSION 3.17)

project(tritonDigiGPUbackend LANGUAGES CUDA C CXX)

option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "r21.02" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r21.02" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r21.02" CACHE STRING "Tag for triton-inference-server/backend repo")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

if (${TRITON_ENABLE_GPU})
    find_package(CUDAToolkit REQUIRED)
endif ()  # TRITON_ENABLE_GPU

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

#
# libDigiGPU.so
#
configure_file(src/libtriton_DigiGPU.ldscript libtriton_DigiGPU.ldscript COPYONLY)

set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
add_library(
  triton-DigiGPU-backend SHARED
  src/DigiGPU.cu
  src/DigiGPU_utils.cc
  src/DigiGPU_utils.h
)

add_library(
  TritonDigiGPUBackend::triton-DigiGPU-backend ALIAS triton-DigiGPU-backend
)

target_include_directories(
  triton-DigiGPU-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(
  triton-DigiGPU-backend
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverstub  # from repo-core
)


set_target_properties(
  triton-DigiGPU-backend PROPERTIES
  OUTPUT_NAME triton_DigiGPU
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_DigiGPU.ldscript
  PROPERTIES LINK_FLAGS "-Wl,--version-script libtriton_DigiGPU.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonDigiGPUBackend)

install(
  TARGETS
    triton-DigiGPU-backend
  EXPORT
    triton-DigiGPU-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/DigiGPU
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/DigiGPU
)

install(
  EXPORT
    triton-DigiGPU-backend-targets
  FILE
    TritonDigiGPUBackendTargets.cmake
  NAMESPACE
    TritonDigiGPUBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonDigiGPUBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonDigiGPUBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonDigiGPUBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-DigiGPU-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonDigiGPUBackendTargets.cmake
  NAMESPACE TritonDigiGPUBackend::
)

export(PACKAGE TritonDigiGPUBackend)

