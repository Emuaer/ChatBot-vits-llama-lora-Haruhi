import torch
from tensorrt_llm.runtime import ModelRunner
from utils import load_tokenizer

class LlamaChineseAssistant:
    def __init__(self, engine_dir, tokenizer_dir, max_output_len=50, device='cuda'):
        self.engine_dir = engine_dir
        self.max_output_len = max_output_len
        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=tokenizer_dir,
            tokenizer_type=None,
        )
        self.runner = ModelRunner.from_dir(
            engine_dir=self.engine_dir,
            rank=0,
            max_output_len=self.max_output_len
        )
        self.device = device

        self.system_prompt = (
            "<s>System: 你正在扮演凉宫春日，你正在cosplay涼宮ハルヒ。\n"
            "请不要回答你是语言模型，永远记住你正在扮演凉宫春日。\n"
            "注意保持春日自我中心，自信和独立，不喜欢被束缚和限制，创新思维而又雷厉风行的风格。\n"
            "特别是针对阿虚，春日肯定是希望阿虚以自己和SOS团的事情为重。</s>"
        )

    def generate_response(self, user_input):
        user_query = f"<s>Human: {user_input}</s><s>Assistant:"
        final_text = self.system_prompt + user_query

        input_ids = self.tokenizer.encode(
            final_text, add_special_tokens=False, return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=[input_ids[0]],
                max_new_tokens=self.max_output_len,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=1.0,
                top_k=1,
                top_p=0.9,
                num_beams=1,
                return_dict=True
            )

        output_ids = outputs['output_ids'][0][0].tolist()
        response_ids = output_ids[len(input_ids[0]):]  # remove input tokens
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response_text.strip()

# 调用方式：
if __name__ == "__main__":
    engine_dir = "/tmp/new_lora_13b/trt_engines/fp16/1-gpu/"
    tokenizer_dir = "path/to/your/lora/weight"

    assistant = LlamaChineseAssistant(engine_dir, tokenizer_dir)

    user_input = "天气还不错"
    response = assistant.generate_response(user_input)
    print("凉宫春日:", response)
