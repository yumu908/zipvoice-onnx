
"""
mkdir -p ./model
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/text_encoder.onnx -O ./model/text_encoder.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/fm_decoder.onnx -O ./model/fm_decoder.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/text_encoder_int8.onnx -O ./model/text_encoder_int8.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/fm_decoder_int8.onnx -O ./model/fm_decoder_int8.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/model.json -O ./model/model.json
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/tokens.txt -O ./model/tokens.txt


mkdir -p ./model_distilled
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/text_encoder.onnx -O ./model_distilled/text_encoder.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/fm_decoder.onnx -O ./model_distilled/fm_decoder.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/text_encoder_int8.onnx -O ./model_distilled/text_encoder_int8.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/fm_decoder_int8.onnx -O ./model_distilled/fm_decoder_int8.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/model.json -O ./model_distilled/model.json
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/tokens.txt -O ./model_distilled/tokens.txt

mkdir -p ./model_vocos
wget https://huggingface.co/wetdog/vocos-mel-24khz-onnx/resolve/main/mel_spec_24khz.onnx -O ./model_vocos/mel_spec_24khz.onnx
wget https://huggingface.co/wetdog/vocos-mel-24khz-onnx/resolve/main/config.yaml -O ./model_vocos/config.yaml


uv run examples/english.py
"""

import soundfile as sf
from zipvoice_onnx import ZipVoice, ZipVoiceOptions

# Example usage with zipvoice_distill model
options = ZipVoiceOptions(
    text_encoder_path="./model_distilled/text_encoder.onnx",
    fm_decoder_path="./model_distilled/fm_decoder.onnx",
    text_encoder_int8_path="./model_distilled/text_encoder_int8.onnx",
    fm_decoder_int8_path="./model_distilled/fm_decoder_int8.onnx",
    model_json_path="./model_distilled/model.json",
    tokens_path="./model_distilled/tokens.txt",
    vocos_model_path="./model_vocos/mel_spec_24khz.onnx",
)

zipvoice = ZipVoice(options)

# Example usage
ref_wav = "prompt.wav"
ref_phonemes = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."
target_phonemes = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."

samples, sample_rate = zipvoice.create(ref_wav, ref_phonemes, target_phonemes, num_steps=32)
print(f"Generated audio: {samples.shape} samples at {sample_rate} Hz")

sf.write("audio.wav", samples, sample_rate)
print("Saved to audio.wav")
