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

## lora权重以及预训练权重下载
> **[lora权重下载](https://pan.baidu.com/s/1MQWZ45OweIcomv5knu6OyQ)**:05wi
> **[Llama3-Chinese-8B-Instruct 权重下载](https://huggingface.co/FlagAlpha/Llama3-Chinese-8B-Instruct)**

## Acknowledgements

* https://github.com/RVC-Boss/GPT-SoVITS
* https://github.com/LlamaFamily/Llama-Chinese
* ...
