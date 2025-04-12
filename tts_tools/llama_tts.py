import requests
import os
from . import wav_text
import time
from baidu_translate.BaiDuTranslate import BaiDuTranslate


def convert_language_code(lang: str):
    lang_map = {
        "中文": "zh",
        "zh": "zh",
        "英文": "en",
        "en": "en",
        "日文": "jp",
        "ja": "jp",
        "韩文": "kor",
        "ko": "kor",
        "粤语": "yue"
    }
    return lang_map.get(lang, "zh")  # 默认返回中文


def translate_text(input_text, target_language,key,appid):
    toLan = convert_language_code(target_language)# 百度api翻译对齐
    translator = BaiDuTranslate(key=key,appid=appid,fromLan='zh',toLan=toLan)
    translate_result=translator.requestApi(query=input_text)
    return translate_result

def process_text_requests(input_text, refer_wav_path, output_folder, api_url,language,target_language,key,api):
    # 解析参考音频文本
    transcriber = wav_text.WhisperTranscriber(
        model_path="/home/chwu/MODELS/whisper-medium",
        language=language
    )
    refer_wav_text = transcriber.transcribe(refer_wav_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 非中文调用百度翻译api
    if target_language!='zh': 
        print(f"📝 原始文本: {input_text}")
        input_text = translate_text(input_text=input_text,
                                    target_language=target_language,
                                    key=key,
                                    appid=api)
        print(f"🌐 翻译后文本: {input_text}")

    # 请求 gpt-sovits 接口合成语音，参数调整可以在此处进行
    payload = {
        "refer_wav_path": refer_wav_path,
        "prompt_text": refer_wav_text,
        "prompt_language": language,
        "text": input_text,
        "text_language": target_language,
        "top_k": 20,
        "top_p": 0.6,
        "temperature": 0.6,
        "speed": 0.9,
        "st": 'int32',
        "mt": "wav"
    }

    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return None

    if response.status_code == 200:
        filename = f"output_{int(time.time())}.wav"
        output_path = os.path.join(output_folder, filename)
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ 音频保存成功: {output_path}")
        return output_path  # 返回音频路径供页面使用
    else:
        print(f"❌ 状态码异常: {response.status_code}, 内容: {response.text}")
        return None

        

if __name__ == "__main__":
    input_text = 'お前のことだから、少しかっこいいなーとか思って言ったのだろう。んで、引くに引けない状態になったんだろ'
    output_folder = './tts_output'
    refer_wav_path='GPT-SoVITS-main/GPT-SoVITS-main/data_row_text/nat/nat002_normal_2_彼女が死神ってことは知ってるけど、私は普通の人間.wav'
    api_url = 'http://localhost:12351/'
    process_text_requests(input_text,refer_wav_path, output_folder, api_url)
