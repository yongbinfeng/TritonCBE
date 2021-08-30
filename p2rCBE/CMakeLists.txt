cmake_minimum_required (VERSION 3.17)

project(tritonp2rbackend LANGUAGES CUDA C CXX)

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

#
# libp2r.so
#
configure_file(src/libtriton_p2r.ldscript libtriton_p2r.ldscript COPYONLY)

set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
add_library(
  triton-p2r-backend SHARED
  src/p2r.cu
  #$<TARGET_OBJECTS:model-config-cuda-library>
)

add_library(
  TritonP2rBackend::triton-p2r-backend ALIAS triton-p2r-backend
)

target_include_directories(
  triton-p2r-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(
  triton-p2r-backend
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverstub  # from repo-core
)


set_target_properties(
  triton-p2r-backend PROPERTIES
  OUTPUT_NAME triton_p2r
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_p2r.ldscript
  PROPERTIES LINK_FLAGS "-Wl,--version-script libtriton_p2r.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonP2rBackend)

install(
  TARGETS
    triton-p2r-backend
  EXPORT
    triton-p2r-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/p2r
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/p2r
)

install(
  EXPORT
    triton-p2r-backend-targets
  FILE
    TritonP2rBackendTargets.cmake
  NAMESPACE
    TritonP2rBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonP2rBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonP2rBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonP2rBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-p2r-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonP2rBackendTargets.cmake
  NAMESPACE TritonP2rBackend::
)

export(PACKAGE TritonP2rBackend)

