# Ensure we are in the project root

# Create the folder (no‑op if it already exists)
mkdir model-en-distilled

# Distilled model files
curl -L -o model-en-distilled/text_encoder_int8.onnx    https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/text_encoder_int8.onnx
curl -L -o model-en-distilled/fm_decoder_int8.onnx     https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/fm_decoder_int8.onnx
curl -L -o model-en-distilled/model.json               https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/model.json
curl -L -o model-en-distilled/tokens.txt               https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/tokens.txt

# Vocoder and prompt audio
curl -L -o vocos_24khz.onnx https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/vocos_24khz.onnx
curl -L -o prompt_english_female1.wav https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/prompt_english_female1.wav
