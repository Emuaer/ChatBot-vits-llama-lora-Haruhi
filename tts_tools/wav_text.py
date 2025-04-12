from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch

class WhisperTranscriber:
    def __init__(self, model_path: str, language: str = "japanese", task: str = "transcribe"):
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)

    def transcribe(self, audio_path: str, sampling_rate: int = 16000) -> str:
        audio, _ = librosa.load(audio_path, sr=sampling_rate)
        input_features = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
        predicted_ids = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]

# 使用示例
if __name__ == "__main__":
    model_path = "/home/chwu/MODELS/whisper-medium"
    audio_path = "/home/chwu/MODELS/GPT-SoVITS-main/GPT-SoVITS-main/data_row_text/nat/nat002_normal_2.wav"

    transcriber = WhisperTranscriber(model_path)
    result = transcriber.transcribe(audio_path)

    print("Transcription result:", result)