"""
See https://github.com/gemelo-ai/vocos/issues/38

wget https://storage.googleapis.com/audioset/speech_whistling2.wav -O speech_whistling2.wav
uv pip install huggingface-hub onnxruntime torch torchaudio torchcodec

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:$DYLD_LIBRARY_PATH"
uv run src/export_vocos.py
"""
from huggingface_hub import hf_hub_download
import onnxruntime
import torch
import torchaudio
import torchaudio.functional as F
from pathlib import Path
import yaml
import shutil


def download_and_save_onnx_model(
    repo_id: str = "wetdog/vocos-mel-24khz-onnx",
    model_filename: str = "mel_spec_24khz.onnx",
    config_filename: str = "config.yaml",
    output_dir: str = "./model_vocos",
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download files
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
    
    # Save to output directory
    output_model_path = output_path / model_filename
    output_config_path = output_path / config_filename
    
    shutil.copy2(model_path, output_model_path)
    shutil.copy2(config_path, output_config_path)
    
    print(f"Model saved to: {output_model_path}")
    print(f"Config saved to: {output_config_path}")
    
    return output_model_path, output_config_path


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    signal, fs = torchaudio.load(audio_path)
    
    if fs != target_sample_rate:
        signal = F.resample(signal, fs, target_sample_rate)
    
    return signal


def extract_mel_spectrogram(signal: torch.Tensor, params: dict) -> torch.Tensor:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=params["sample_rate"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        n_mels=params["n_mels"],
        center=params["padding"] == "center",
        power=1,
    )
    
    mel = torch.log(torch.clip(mel_transform(signal), min=1e-7))
    return mel


def load_onnx_model(model_path: Path, providers: list = None) -> onnxruntime.InferenceSession:
    if providers is None:
        providers = ["CPUExecutionProvider"]
    
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers
    )
    return model


def run_vocos_inference(model: onnxruntime.InferenceSession, mel: torch.Tensor) -> torch.Tensor:
    mag, x, y = model.run(
        None,
        {"mels": mel.float().numpy()},
    )
    
    # Complex spectrogram from vocos output
    spectrogram = mag * (x + 1j * y)
    
    return torch.tensor(spectrogram)


def reconstruct_audio(spectrogram: torch.Tensor, params: dict) -> torch.Tensor:
    win_length = params["n_fft"]
    audio = torch.istft(
        spectrogram,
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=True
    )
    return audio


def process_audio_with_vocos(
    audio_input: str,
    model_path: Path,
    config_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = load_config(config_path)
    params = config["feature_extractor"]["init_args"]
    
    # Load audio
    signal = load_audio(audio_input, params["sample_rate"])
    
    # Extract mel spectrogram
    mel = extract_mel_spectrogram(signal, params)
    
    # Load and run ONNX model
    model = load_onnx_model(model_path)
    spectrogram = run_vocos_inference(model, mel)
    
    # Reconstruct audio
    audio = reconstruct_audio(spectrogram, params)
    
    return signal, audio


if __name__ == "__main__":
    # Download and save ONNX model
    model_path, config_path = download_and_save_onnx_model()
    
    # Process audio
    audio_input = "speech_whistling2.wav"
    signal, audio = process_audio_with_vocos(audio_input, model_path, config_path)
    
    # print("Original audio")
    # display(Audio(data=signal, rate=params["sample_rate"]))
    # print("Vocos reconstruction")
    # display(Audio(data=audio, rate=params["sample_rate"]))