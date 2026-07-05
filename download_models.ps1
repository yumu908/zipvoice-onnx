# PowerShell script to download ZipVoice distilled model files and supporting assets
# Ensure the directory exists
New-Item -ItemType Directory -Force -Path "model-en-distilled" | Out-Null

# Define URLs and target paths
$downloads = @(
    @{Url = "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/text_encoder.onnx"; Out = "model-en-distilled\text_encoder.onnx"},
    @{Url = "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/fm_decoder.onnx"; Out = "model-en-distilled\fm_decoder.onnx"},
    @{Url = "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/text_encoder_int8.onnx"; Out = "model-en-distilled\text_encoder_int8.onnx"},
    @{Url = "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/fm_decoder_int8.onnx"; Out = "model-en-distilled\fm_decoder_int8.onnx"},
    @{Url = "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/model.json"; Out = "model-en-distilled\model.json"},
    @{Url = "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/tokens.txt"; Out = "model-en-distilled\tokens.txt"},
    @{Url = "https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/vocos_24khz.onnx"; Out = "vocos_24khz.onnx"},
    @{Url = "https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/prompt_english_female1.wav"; Out = "prompt_english_female1.wav"}
)

foreach ($item in $downloads) {
    Write-Host "Downloading $($item.Url) -> $($item.Out)"
    Invoke-WebRequest -Uri $item.Url -OutFile $item.Out -UseBasicParsing
}

Write-Host "All downloads completed."
