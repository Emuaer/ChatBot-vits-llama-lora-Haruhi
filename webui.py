import gradio as gr
import os
from tts_tools import llama_tts
from chat_tools import llama_haruhi_lora

def run_chatbot(
    query: str,
    history: list,
    base_model: str,
    lora_model: str,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    refer_wav_path: str,
    output_folder: str,
    api_url: str,
    language_box:str,
    target_language_box:str,
    key_box:str,
    api_box:str
):
    max_new_tokens = int(max_new_tokens)
    top_k = int(top_k)
    top_p = float(top_p)
    temperature = float(temperature)
    repetition_penalty = float(repetition_penalty)
    language=language_box
    target_language=target_language_box
    key=key_box
    api=api_box

    # 调用你的对话生成函数
    response_text = llama_haruhi_lora.generate_haruhi_response(
        base_model=base_model,
        lora_model=lora_model,
        query=query,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    # 语音合成
    audio_path = llama_tts.process_text_requests(
        input_text=response_text,
        refer_wav_path=refer_wav_path,
        output_folder=output_folder,
        api_url=api_url,
        language=language,
        target_language=target_language,
        key=key,
        api=api
    )

    # 更新对话记录
    history.append((query, response_text))
    return history, gr.update(value=audio_path)

def build_gradio_ui():
    with gr.Blocks(title="HaruhiBot") as demo:
        gr.Markdown(
            """
            # 🌟  HaruhiBot
            你可以和凉宫春日聊天，她会用语音回应你！
            输入任何你想说的话，她会回复你并语音合成。
            """
        )

        # 将「模型与音频设置」和「生成参数」并列放置
        with gr.Row():
            # 左侧：模型与音频设置
            with gr.Column(scale=1):
                gr.Markdown("## 模型与音频设置")
                base_model_box = gr.Textbox(
                    label="基础模型路径",
                    value="path/to/your/base_model"
                )
                lora_model_box = gr.Textbox(
                    label="LoRA微调模型路径",
                    value="path/to/your/lora_model"
                )
                refer_wav_box = gr.Textbox(
                    label="参考音频路径",
                    value="./nat002_normal.wav"
                )
                output_folder_box = gr.Textbox(
                    label="生成音频保存文件夹",
                    value="./tts_output"
                )
                api_url_box = gr.Textbox(
                    label="gpt-sovits的api地址",
                    value="http://localhost:12351/"
                )
                key_box = gr.Textbox(
                    label="百度翻译api的key",
                     value="申请地址 https://fanyi-api.baidu.com/"
                )
                api_box = gr.Textbox(
                    label="百度翻译api的key",
                     value="申请地址 https://fanyi-api.baidu.com/"
                )


            # 右侧：生成参数
            with gr.Column(scale=1):
                gr.Markdown("## 生成参数")
                max_new_tokens_slider = gr.Slider(
                    minimum=1,
                    maximum=2048,
                    step=1,
                    value=512,
                    label="max_new_tokens"
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=50,
                    label="top_k"
                )
                top_p_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.95,
                    label="top_p"
                )
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=0.3,
                    label="temperature"
                )
                repetition_penalty_slider = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    step=0.1,
                    value=1.3,
                    label="repetition_penalty"
                )
                language_box = gr.Dropdown(
                    label="参考音频文本语种",
                    choices=["zh", "en", "ja"],
                    value="ja",  # 默认选项
                    interactive=True
                )
                target_language_box = gr.Dropdown(
                    label="回复音频文本语种",
                    choices=["zh", "en", "ja"],
                    value="ja",  # 默认选项
                    interactive=True
                )

        # 对话框区域
        chatbot = gr.Chatbot(label="🗣️ 对话框", bubble_full_width=False)
        with gr.Row():
            with gr.Column(scale=4):
                user_input = gr.Textbox(
                    label="你要说的话",
                    placeholder="比如：请介绍一下你自己！",
                    show_label=False
                )
            with gr.Column(scale=1):
                send_btn = gr.Button("发送")

        # 音频输出
        audio_output = gr.Audio(
            label="🎧 Voice",
            type="filepath",
            interactive=False
        )

        # 用于存储历史对话内容
        state = gr.State([])

        # 绑定事件：点击按钮 或 回车时 调用同一个函数
        send_btn.click(
            fn=run_chatbot,
            inputs=[
                user_input,
                state,
                base_model_box,
                lora_model_box,
                max_new_tokens_slider,
                top_k_slider,
                top_p_slider,
                temperature_slider,
                repetition_penalty_slider,
                refer_wav_box,
                output_folder_box,
                api_url_box,
                language_box,
                target_language_box,
                key_box,
                api_box
            ],
            outputs=[chatbot, audio_output]
        )

        user_input.submit(
            fn=run_chatbot,
            inputs=[
                user_input,
                state,
                base_model_box,
                lora_model_box,
                max_new_tokens_slider,
                top_k_slider,
                top_p_slider,
                temperature_slider,
                repetition_penalty_slider,
                refer_wav_box,
                output_folder_box,
                api_url_box,
                language_box,
                target_language_box,
                key_box,
                api_box
            ],
            outputs=[chatbot, audio_output]
        )

    return demo

if __name__ == "__main__":
    ui = build_gradio_ui()
    ui.launch(share=False)
