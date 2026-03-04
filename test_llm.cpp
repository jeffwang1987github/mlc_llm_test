//#include <winsock2.h>
#include <stdio.h>
#include <bits/stdc++.h>
#include <iostream>

// https://peppy-fenglisu-c50f28.netlify.app/deploy/cli
#define TVM_USE_LIBBACKTRACE 0
#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/tensor.h>
// #include <tvm/runtime/registry.h>
#include <nlohmann/json.hpp> // 使用 nlohmann/json 库来解析 JSON 文件

#include "tokenizers.h"
#include <tokenizers_cpp.h>


// DLPack is a widely adopted in-memory representation of tensors in deep learning.
#include <dlpack/dlpack.h>

#include <cuda_fp16.h> // CUDA 提供的 float16 支持

#include "llm_chat.h"
#include "conversation.h"

#include "load_bytes_from_file.h"

// using namespace cv;
using namespace std;
using tvm::ffi::Function;
using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::runtime::Tensor;

int main(int argc, char **argv)
{
    const DLDeviceType& device_type = kDLCUDA; // from dlpack.h  用cuda进行推理
    DLDevice device_s = {kDLCUDA, 0};
    int device_id = 0; // which one if there are multiple devices, usually 0

    // // tvm16量化的模型 在这里会报错
    // const std::string& path_model_lib = "/home/wangqx/work/model/VILA1.5-3b/llm_q4f16/VILA1.5-3b-q4f16_ft/VILA1.5-3b-q4f16_ft-cuda.so";
    // const std::string& path_weight_config = "/home/wangqx/work/model/VILA1.5-3b/llm_q4f16/VILA1.5-3b-q4f16_ft/params/";//只是路进，不含文件名

    // //tvm23 python直接量化的模型
    // const std::string& path_model_lib = "/home/wangqx/work/model/VILA1.5-3b/llm_q4f16_tvm_23/VILA1.5-3b-q4f16_ft-cuda.so";
    // const std::string& path_weight_config = "/home/wangqx/work/model/VILA1.5-3b/llm_q4f16_tvm_23/params/";//只是路进，不含文件名

    // qwen3 
    const std::string& path_model_lib = "/home/wangqx/work/model/qwen3/q4f16_23/qwen3-2b-q4f16_ft-cuda.so";
    const std::string& path_weight_config = "/home/wangqx/work/model/qwen3/q4f16_23/params/";//只是路进，不含文件名

   using tvm::ffi::Function;

   int64_t max_window_size = 8192; // 设置KV cache大小，单位为token数量

   tvm::ffi::Module mlc_llm =  mlc::llm::CreateChatModule(device_s);

   // // Step 2. Obtain all available functions in `mlc_llm`
   Function prefill = mlc_llm->GetFunction("prefill").value();

   Function decode = mlc_llm->GetFunction("decode").value();
   Function stopped = mlc_llm->GetFunction("stopped").value();
   Function get_message = mlc_llm->GetFunction("get_message").value();
   Function reload = mlc_llm->GetFunction("reload").value();

   Function embedding = mlc_llm->GetFunction("embed").value();
   Function prefill_with_embed = mlc_llm->GetFunction("prefill_with_embed").value();
   Function get_role0 = mlc_llm->GetFunction("get_role0").value();
   Function get_role1 = mlc_llm->GetFunction("get_role1").value();
   Function runtime_stats_text = mlc_llm->GetFunction("runtime_stats_text").value();
   Function reset_chat = mlc_llm->GetFunction("reset_chat").value();
   Function raw_generate = mlc_llm->GetFunction("raw_generate").value();
   Function unload = mlc_llm->GetFunction("unload").value(); //清除内存
   Function process_system_prompts = mlc_llm->GetFunction("process_system_prompts").value();
   Function resize_kv_cache = mlc_llm->GetFunction("resize_kv_cache").value(); //动态调整KV cache
   // Step 3. Load the model lib containing optimized tensor computation
   static tvm::ffi::Module model_lib = tvm::ffi::Module::LoadFromFile(path_model_lib);

   // Step 4. Reload model
   // KV cache size is controlled by max_window_size in mlc-chat-config.json
   // Can be adjusted dynamically using resize_kv_cache() function
   std::cout << "===== Calling reload =====" << std::endl;
   reload(model_lib, path_weight_config, "",max_window_size);
   std::cout << "===== Reload completed =====" << std::endl;

   // Step 5. 重置聊天状态
   reset_chat();

   std::string system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n";
   std::string input_message;
   std::vector<std::string> history; // 用于存储历史对话
//    const size_t MAX_HISTORY_TURNS = 5; // 限制历史对话轮数以减少GPU KV cache占用

    //测试先embedding然后prefill_with_embed
    // 输入 token IDs

    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, input_message);

        if (input_message == "exit") {
            unload(); // 卸载模型
            std::cout << "Exiting the chat. Goodbye!" << std::endl;
            break; // 输入 "exit" 退出对话
        }

        if(input_message.empty()) {
            std::cout << "Input message cannot be empty. Please try again." << std::endl;
            continue; // 如果输入为空，提示用户并继续循环
        }

        if(input_message == "reset") {
            reset_chat(); // 重置聊天状态
            history.clear(); // 清空历史对话
            std::cout << "Chat history has been reset." << std::endl;
            continue; // 如果输入为 "reset"，重置聊天状态并继续循环
        }

        // Dynamic KV cache resize command: "resize:4096"
        if(input_message.substr(0, 7) == "resize:") {
            try {
                int64_t new_size = std::stoll(input_message.substr(7));
                std::cout << "Resizing KV cache to " << new_size << " tokens..." << std::endl;
                resize_kv_cache(new_size);
                history.clear(); // Clear history after resize
                std::cout << "KV cache resized. Chat history cleared." << std::endl;
            } catch (...) {
                std::cout << "Invalid resize command. Usage: resize:4096" << std::endl;
            }
            continue;
        }

        // 拼接历史对话
        std::string full_context = system_prompt;
        for (const auto& msg : history) {
            full_context += msg + "\n";
        }
        full_context += "USER: " + input_message + "\nASSISTANT:";
        std::cout << "the input string is: " << full_context << std::endl;

        // Step 7. 执行推理

        // //7.1 调用 prefill 函数
        // prefill(full_context); // 预填充输入
        //7.2 先调用embedding再调用prefill (使用局部变量避免内存累积)
        {
            Tensor embedding_result = embedding(full_context).as<Tensor>().value();
            prefill_with_embed(embedding_result);
        }
        //7.3 显存copy到cpu再推理 注意传递编码时一定要copy到GPU
        // embedding_result = embedding(full_context);
        // Tensor cpu_array = embedding_result.CopyTo({kDLCPU, 0});//要先将数据copy到CPU，然后再操作
        // Tensor gpu_array = cpu_array.CopyTo({kDLCUDA, 0});//要先将数据copy到CPU，然后再操作
        // prefill_with_embed(gpu_array);

        while (!stopped().as<bool>().value()) {
            decode(); // 解码
        }

        // 获取模型生成的响应
        std::string output_message = get_message().as<String>().value();
        std::cout << "Assistant: " << output_message << std::endl;

        // 将当前对话加入历史
        history.push_back("USER: " + input_message);
        history.push_back("ASSISTANT: " + output_message);

        // 限制历史对话数量，超过限制则删除最旧的对话以减少GPU内存占用
        // if (history.size() > MAX_HISTORY_TURNS * 2) {
        //     history.erase(history.begin(), history.begin() + 2);
        // }
    }

    return 0;
}