import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# 仅初始化一次模型和tokenizer（建议全局执行一次）
device_map = "cuda:0" if torch.cuda.is_available() else "auto"

def generate_haruhi_model_tokenizer(finetune_model_path, base_model_name_or_path):
    config = PeftConfig.from_pretrained(finetune_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map=device_map,
                                                torch_dtype=torch.float16, load_in_8bit=True,
                                                trust_remote_code=True, use_flash_attention_2=True)
    model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
    model = model.eval()
    return model, tokenizer

def generate_extracted(start_tag,end_tag,raw_output):
    # 提取回答的内容
    start_idx = raw_output.find(start_tag)
    end_idx = raw_output.find(end_tag, start_idx)
    if start_idx != -1 and end_idx != -1:
        # 提取 Assistant 的实际内容并去除多余的空白
        extracted = raw_output[start_idx + len(start_tag):end_idx].strip()
    else:
        extracted = "[Warning] Assistant response not found."
    return extracted

def generate_haruhi_response(base_model, lora_model, query,max_new_tokens,top_k,top_p,temperature,repetition_penalty) -> str:
    # 系统提示词，根据下游具体lora数据集决定
    system_prompt = (
        """<s>System: 你正在扮演凉宫春日，你正在cosplay涼宮ハルヒ。
请不要回答你是语言模型，永远记住你正在扮演凉宫春日。
注意保持春日自我中心，自信和独立，不喜欢被束缚和限制，创新思维而又雷厉风行的风格。
特别是针对阿虚，春日肯定是希望阿虚以自己和SOS团的事情为重。</s>"""
    )
    
    # 拼接完整 prompt
    query = f"<s>Human: {query}</s><s>Assistant:"
    input_text = system_prompt + query
    model, tokenizer = generate_haruhi_model_tokenizer(lora_model, base_model)
    
    # 编码
    input_ids = tokenizer([input_text], return_tensors="pt", add_special_tokens=False).input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    # 推理参数
    generate_input = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_k":top_k,
        "top_p":top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }

    # 生成 & 解码
    generate_ids = model.generate(**generate_input)
    # 解码文本
    raw_output = tokenizer.decode(generate_ids[0], skip_special_tokens=False)

    # 提取回答的内容
    start_tag = "<s>Assistant:"
    end_tag = "</s>"

    extracted=generate_extracted(start_tag=start_tag,end_tag=end_tag,raw_output=raw_output)

    # 输出
    # print("\n🧑 user:", user_query)
    # print("\n🗣️ 凉宫春日:", extracted)

    return extracted

def generate_haruhi_response_local(user_query, finetune_model_path, base_model_name_or_path) -> str:
    # 系统提示词，根据下游具体lora数据集决定
    system_prompt = (
        """<s>System: 你正在扮演凉宫春日，你正在cosplay涼宮ハルヒ。
请不要回答你是语言模型，永远记住你正在扮演凉宫春日。
注意保持春日自我中心，自信和独立，不喜欢被束缚和限制，创新思维而又雷厉风行的风格。
特别是针对阿虚，春日肯定是希望阿虚以自己和SOS团的事情为重。</s>"""
    )
    
    # 拼接完整 prompt
    query = f"<s>Human: {user_query}</s><s>Assistant:"
    input_text = system_prompt + query
    model, tokenizer = generate_haruhi_model_tokenizer(finetune_model_path, base_model_name_or_path)
    
    # 编码
    input_ids = tokenizer([input_text], return_tensors="pt", add_special_tokens=False).input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    # 推理参数
    generate_input = {
        "input_ids": input_ids,
        "max_new_tokens": 512,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.3,
        "repetition_penalty": 1.3,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }

    # 生成 & 解码
    generate_ids = model.generate(**generate_input)
    # 解码文本
    raw_output = tokenizer.decode(generate_ids[0], skip_special_tokens=False)

    # 提取回答的内容
    start_tag = "<s>Assistant:"
    end_tag = "</s>"

    start_idx = raw_output.find(start_tag)
    end_idx = raw_output.find(end_tag, start_idx)

    if start_idx != -1 and end_idx != -1:
        # 提取 Assistant 的实际内容并去除多余的空白
        extracted = raw_output[start_idx + len(start_tag):end_idx].strip()
    else:
        extracted = "[Warning] Assistant response not found."

    # 输出
    print("\n🧑 user:", user_query)
    print("\n🗣️ 凉宫春日:", extracted)

    return extracted

if __name__ == "__main__":
    finetune_model_path = '/home/chwu/MODELS/Llama-Chinese-main/Llama-Chinese-main/save_folder'  
    base_model_name_or_path = '/home/chwu/MODELS/Llama3-Chinese-8B-Instruct'
    
    # 示例用户输入
    user_input = "我喜欢吃炸鸡，你想吃吗"
    
    # 获取并打印结果
    result_msg = generate_haruhi_response_local(user_input, finetune_model_path, base_model_name_or_path)
