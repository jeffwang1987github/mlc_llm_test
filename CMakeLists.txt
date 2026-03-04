
cmake_minimum_required(VERSION 3.10)
project(test C CXX)

set(CMAKE_CXX_FLAGS "-O3")

# 用c++17 
set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_BUILD_TYPE "Debug")
#设置可执行文件输出目录
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

# Set OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")


# Set Eigen3 for MOT
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")

# # Set TVM lib path
set(TVM_DIR ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/3rdparty/tvm)
include_directories(${TVM_DIR}/include)
link_directories(${TVM_DIR}/lib)

# Set CUDA path
set(CUDA_DIR /usr/local/cuda)
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)
list(APPEND CUDA_LIBS cudart)
add_definitions("-DENABLE_FP8")

# Set tensorrt path
set(TensorRT_DIR /usr/src/tensorrt)  # Orin path
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
    set(TensorRT_DIR /usr/src/tensorrt)
    add_definitions("-DARCH_X86")
endif()
include_directories(${TensorRT_DIR}/include ${TensorRT_DIR}/samples ${TensorRT_DIR}/samples/common)
list(APPEND TRT_LIBS nvinfer nvinfer_plugin nvonnxparser)

# Set mlc_llm path
set(MLC_LLM_PATH 3rdparty/mlc_llm)
add_subdirectory(${MLC_LLM_PATH} mlc_llm EXCLUDE_FROM_ALL)

# Add tvm-ffi
add_subdirectory(${TVM_DIR}/3rdparty/tvm-ffi tvm_ffi EXCLUDE_FROM_ALL)


# 强制使用g++编译器编译部分.c文件
set_source_files_properties(
  ${CMAKE_SOURCE_DIR}/src/cjson/cJSON.c
    PROPERTIES LANGUAGE CXX
)


# # 添加所有源文件 会自动递归添加
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/tokenizers/*.cpp
                          ${CMAKE_SOURCE_DIR}/src/siglip/*.cpp
                          ${CMAKE_SOURCE_DIR}/src/utils/*.cpp
                          ${CMAKE_SOURCE_DIR}/src/det/*.cpp
                          ${CMAKE_SOURCE_DIR}/src/vlm/*.cpp
                          # ${CMAKE_SOURCE_DIR}/src/cjson/*.c
                          )

# 单独添加编译文件 要在上一个添加之后
list(APPEND SOURCES
    ${CMAKE_SOURCE_DIR}/src/cjson/cJSON.c
    )

# 添加头文件
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/3rdparty/tvm/3rdparty/dlpack/include
                    ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/3rdparty/tvm/3rdparty/dmlc-core/include
                    ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/3rdparty/tvm/3rdparty/picojson
                    ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/3rdparty/tvm/3rdparty/tvm-ffi/include
                    ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include

                    # ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/3rdparty/tvm/3rdparty/dmlc-core/include
                    # ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/3rdparty/tvm/3rdparty/picojson

                    ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/3rdparty/tokenizers-cpp/include
                    ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/src
                    ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/src/tokenizers
                    ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/src/support
                    ${CMAKE_SOURCE_DIR}/3rdparty/mlc_llm/src/json_ffi
                    ${CMAKE_SOURCE_DIR}/src/cjson
                    )

add_executable(test ${CMAKE_SOURCE_DIR}/test_llm.cpp ${SOURCES})

target_link_libraries(test PRIVATE ${OPENCV_LIBS} ${CUDA_LIBS} ${OpenCV_LIBS} ${TRT_LIBS} Eigen3::Eigen
                        -lrt -lpthread -lavcodec -lavformat -lavutil -lswscale
                         -lflash_attn -lfpA_intB_gemm
                        -ltvm
                        tvm_ffi_shared
                        mlc_llm #依赖mlc_llm的子项目
                        )