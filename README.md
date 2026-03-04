# MLC-LLM C++ Inference

基于 [MLC-LLM](https://github.com/mlc-ai/mlc-llm) 和 [TVM](https://tvm.apache.org/) 的 C++ 大语言模型推理框架，支持在 NVIDIA GPU 上高效运行量化 LLM 模型（如 Qwen、VILA 等），提供交互式对话功能。

## 特性

- GPU 加速推理（CUDA + TensorRT）
- 支持 Q4F16 量化模型，降低显存占用
- 支持 FP8 精度推理
- 交互式 CLI 对话界面，支持多轮对话历史
- 动态 KV Cache 大小调整
- 支持视觉语言模型（VLM）
- 支持 x86_64 和 ARM64（Jetson Orin）平台

## 系统要求

- NVIDIA GPU（Compute Capability 7.0+）
- CUDA Toolkit（11.0+）
- TensorRT
- CMake 3.10+
- C++17 编译器

## 依赖项

| 依赖 | 用途 |
|------|------|
| TVM | 张量计算运行时 |
| CUDA | GPU 计算 |
| TensorRT | 推理优化 |
| OpenCV | 图像处理（VLM） |
| Eigen3 | 线性代数 |
| FFmpeg (libavcodec 等) | 多媒体处理 |
| Flash Attention | 注意力机制加速 |

> 项目中已包含 MLC-LLM、TVM、tokenizers-cpp 等第三方源码（位于 `3rdparty/` 目录）。

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

编译完成后，可执行文件 `test` 将生成在项目根目录。

## 模型准备

需要预先使用 MLC-LLM 工具链将模型编译为 CUDA 共享库和参数文件。编译完成后修改 `test_llm.cpp` 中的路径：

```cpp
// 模型库文件（.so）
const std::string& path_model_lib = "/path/to/model-cuda.so";
// 模型参数目录
const std::string& path_weight_config = "/path/to/params/";
```

目前已测试的模型：
- Qwen3-2B (Q4F16)
- VILA 1.5-3B (Q4F16)

## 使用

```bash
./test
```

启动后进入交互式对话：

```
You: 你好，请介绍一下自己
Assistant: 你好！我是一个人工智能助手...
```

支持的命令：

| 命令 | 说明 |
|------|------|
| `exit` | 退出程序 |
| `reset` | 重置对话历史 |
| `resize:<N>` | 动态调整 KV Cache 大小（如 `resize:4096`） |

## 项目结构

```
.
├── test_llm.cpp           # 主程序入口
├── CMakeLists.txt         # 构建配置
├── src/
│   ├── cjson/             # JSON 解析
│   ├── tokenizers/        # 分词器
│   ├── siglip/            # 视觉编码器
│   ├── vlm/               # 视觉语言模型
│   ├── det/               # 目标检测
│   └── utils/             # 工具函数
└── 3rdparty/
    └── mlc_llm/           # MLC-LLM 框架
        ├── src/           # 核心推理代码
        └── 3rdparty/      # TVM、tokenizers-cpp 等
```

## License

[MIT](LICENSE) - Copyright (c) 2026 jeffwang1987github
