import torch
from transformers import AutoModelForCausalLM

base_model_name_or_path = 'Llama3-Chinese-8B-Instruct'

# 加载基础模型 (fp16)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)


# 保存合并后的模型
model.save_pretrained('./Merged_Lora_Model')