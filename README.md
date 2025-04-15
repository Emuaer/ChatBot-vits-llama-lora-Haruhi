# 🦙🌟 ChatBot-vits-llama-lora-Haruhi

 > 基于 `Llama-Chinese` lora微调 和 `GPT-SoVITS`的实时语音聊天Bot！

> **Update**:
> 
> 上传了初始版本， 其中Haruhi的数据集语料来自：**[Chat-Haruhi-Suzumiya](https://github.com/LC1332/Chat-Haruhi-Suzumiya?tab=readme-ov-file)** 
> 
> 语音生成API来自：**[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)** 
>
>对话模型来自：**[Llama-Chinese](https://github.com/LlamaFamily/Llama-Chinese)** 进行lora微调后得到
>

## 功能

**[点击此处跳转Bilibili演示视频](https://huggingface.co/spaces/zetavg/LLaMA-LoRA-UI-Demo)** 
> 与虚拟角色进行实时的交流，并实时生成语音回复！

## 如何开始

首先需要前置配置Llama-Chinese与GPT-Sovits:

* **[启动GPT-Sovits的API](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/README.md)**: 环境安装完成后，需要运行[api.py](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/api.py)启动项目的API以生成语音
* **[克隆Llama-Chinese至本地](#https://github.com/LlamaFamily/Llama-Chinese)**: 运行pip install -r requirements.txt 安装依赖



### 依赖安装

<details>
  <summary>在conda中创建虚拟环境</summary>

  ```bash
  conda create -y python=3.10.8 -n llama-bot-chat
  conda activate llama-bot-chat
  ```
</details>

```bash
pip install -r requirements.txt
```
### 运行webui

`python webui.py`.

### 命令行使用
` python main.py --query 今天打算去吃汉堡 --base_model Llama3-Chinese-8B-Instruct --lora_model Llama-Chinese-main/Llama-Chinese-main/save_folder --is_transform True `

## 模型量化
1. 可以使用AutoGPTQ在进行lora微调前进行模型量化，具体可以参考[使用-peft-微调量化后的模型](https://huggingface.co/blog/zh/gptq-integration#--%E4%BD%BF%E7%94%A8-peft-%E5%BE%AE%E8%B0%83%E9%87%8F%E5%8C%96%E5%90%8E%E7%9A%84%E6%A8%A1%E5%9E%8B--)
2. 使用TensorRT-LLM[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM.git)进行推理加速，本项目基于llama，所以可以直接在[链接](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)中参考优化代码`可能会出现降智现象`，示例步骤(单GPU)：

```bash
python convert_checkpoint.py   --model_dir /home/chwu/MODELS/Llama3-Chinese-8B-Instruct   --output_dir ./tllm_checkpoint_1gpu   --dtype float16   --tp_size 1 # Llama权重转换为tensorRT-llm格式
```
```bash
trtllm-build   --checkpoint_dir ./tllm_checkpoint_1gpu   --output_dir /tmp/new_lora_13b/trt_engines/fp16/1-gpu/   --gemm_plugin auto   --lora_plugin auto   --max_batch_size 1   --max_input_len 512   --max_seq_len 562   --lora_dir /path/to/your/lora/weight  --world_size 1 # 生成tensorRT-llm engine
```
```bash
mpirun --allow-run-as-root -n 1 python ../run_wrapper.py   --engine_dir "/tmp/new_lora_13b/trt_engines/fp16/1-gpu/"   --max_output_len 50   --tokenizer_dir /path/to/your/lora/weight   --input_text "介绍一下你自己！"   --lora_task_uids 0   --no_add_special_tokens   --use_py_session # 进行推理
```
```bash
# 也可以通过运行打包文件进行推理
python ChatBot-vits-llama-lora-Haruhi/inference-speed/GPU/TensorRT-LLM_example/run_lora.py
```
## lora权重以及预训练权重下载
> **[lora权重下载](https://pan.baidu.com/s/1MQWZ45OweIcomv5knu6OyQ)**:05wi
> **[Llama3-Chinese-8B-Instruct 权重下载](https://huggingface.co/FlagAlpha/Llama3-Chinese-8B-Instruct)**

## Acknowledgements

* https://github.com/RVC-Boss/GPT-SoVITS
* https://github.com/LlamaFamily/Llama-Chinese
* ...
