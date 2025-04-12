from tts_tools import llama_tts
from chat_tools import llama_haruhi_lora
import argparse
import asyncio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haruhibot Inference Script")
    parser.add_argument("--query", type=str, required=True, help="用户输入内容，例如：介绍一下你自己")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_model", type=str, required=True, help="LoRA微调模型路径")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--load_in_8bit", action="store_true", help="是否使用8bit加载模型")
    parser.add_argument("--refer_wav_path", default='/home/chwu/MODELS/GPT-SoVITS-main/GPT-SoVITS-main/data_row_text/nat/nat002_normal.wav', help="参考音频，后续将加上情感分析")
    parser.add_argument("--language", type=str, default='japanese', help="设定参考音频的语言种类")
    parser.add_argument("--output_folder", default='../tts_output', help="生成音频保存的文件夹")
    parser.add_argument("--is_transform", action="store_true", help="是否开启语音翻译功能(主要看gpt-sovits对应权重效果)")
    parser.add_argument("--target_language", default="ja", help="""开启翻译功能后的目标语种
                        "中文","英文","日文","韩文","zh","en","ja","ko"
                        """)
    parser.add_argument("--key", default='7snDQPIBQY54SBYzJj0r', help="百度翻译的key")
    parser.add_argument("--appid", default='20250411002329939', help="百度翻译的appid")
    parser.add_argument("--api_url", default='http://localhost:12351/', help="gpt-sovits的api地址")

    args = parser.parse_args()
    # 调用llama获得回答
    res_text=llama_haruhi_lora.generate_haruhi_response(args,args.query,args.lora_model,args.base_model)
    # 调用gpt-sovits获得对应语音
    llama_tts.process_text_requests(args,res_text,args.refer_wav_path,args.output_folder,args.api_url)
