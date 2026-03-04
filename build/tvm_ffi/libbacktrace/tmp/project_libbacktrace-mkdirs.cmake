# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/wangqx/work/mlc_llm_test/3rdparty/mlc_llm/3rdparty/tvm/3rdparty/tvm-ffi/cmake/Utils/../../3rdparty/libbacktrace"
  "/home/wangqx/work/mlc_llm_test/build/tvm_ffi/libbacktrace"
  "/home/wangqx/work/mlc_llm_test/build/tvm_ffi/libbacktrace"
  "/home/wangqx/work/mlc_llm_test/build/tvm_ffi/libbacktrace/tmp"
  "/home/wangqx/work/mlc_llm_test/build/tvm_ffi/libbacktrace/src/project_libbacktrace-stamp"
  "/home/wangqx/work/mlc_llm_test/build/tvm_ffi/libbacktrace/src"
  "/home/wangqx/work/mlc_llm_test/build/tvm_ffi/libbacktrace/logs"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/wangqx/work/mlc_llm_test/build/tvm_ffi/libbacktrace/src/project_libbacktrace-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/wangqx/work/mlc_llm_test/build/tvm_ffi/libbacktrace/src/project_libbacktrace-stamp${cfgdir}") # cfgdir has leading slash
endif()
