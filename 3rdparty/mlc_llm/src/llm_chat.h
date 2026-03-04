/*!
 *  Copyright (c) 2023 by Contributors
 * \file llm_chat.h
 * \brief Implementation of llm chat.
 */
// #include <tvm/runtime/container/string.h> //tvm21之前版本
#include <tvm/ffi/string.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/runtime/module.h>

#include "base.h"

namespace mlc {
namespace llm {

// explicit export via TVM_DLL
MLC_LLM_DLL std::string GetDeltaMessage(std::string curr_message, std::string new_message);

// MLC_LLM_DLL tvm::runtime::Module CreateChatModule(DLDevice device); //tvm21之前版本
MLC_LLM_DLL tvm::ffi::Module CreateChatModule(DLDevice device);

}  // namespace llm
}  // namespace mlc
