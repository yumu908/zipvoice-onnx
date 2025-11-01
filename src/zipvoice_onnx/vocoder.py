import numpy as np
import onnxruntime as ort
import yaml
import librosa
from pathlib import Path


def hann_window(n: int) -> np.ndarray:
    if n < 2:
        return np.ones(n, dtype=np.float32)
    return np.hanning(n).astype(np.float32)


class OnnxVocoder:
    def __init__(self, model_path: str = "./model_vocos/mel_spec_24khz.onnx",
                 config_path: str = "./model_vocos/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.params = config["feature_extractor"]["init_args"]
        
        sess_options = ort.SessionOptions()
        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        
        self.win_length = self.params["n_fft"]
        self.hop_length = self.params["hop_length"]
        self.n_fft = self.params["n_fft"]
        self.window = hann_window(self.win_length)
    
    def eval(self):
        pass
    
    def decode(self, mel: np.ndarray) -> np.ndarray:
        mag, x, y = self.model.run(
            None,
            {"mels": mel.astype(np.float32)},
        )
        
        spectrogram = mag * (x + 1j * y)
        
        audio_list = []
        for b in range(spectrogram.shape[0]):
            audio_b = librosa.istft(
                spectrogram[b],
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
            )
            audio_list.append(audio_b)
        
        audio = np.stack(audio_list, axis=0)
        audio = np.expand_dims(audio, axis=1)
        
        return audio


def get_vocoder(vocos_local_path=None):
    if vocos_local_path and str(vocos_local_path).strip():
        vocos_path = Path(vocos_local_path)
        if vocos_path.exists() and vocos_path.is_file():
            model_path = vocos_path
            config_path = vocos_path.parent / "config.yaml"
        else:
            model_path = vocos_path / "mel_spec_24khz.onnx"
            config_path = vocos_path / "config.yaml"
    else:
        model_path = "./model_vocos/mel_spec_24khz.onnx"
        config_path = "./model_vocos/config.yaml"
    
    return OnnxVocoder(str(model_path), str(config_path))


class VocosFbank:
    def __init__(self, sampling_rate: int = 24000, n_mels: int = 100, 
                 n_fft: int = 1024, hop_length: int = 256):
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        assert sampling_rate == self.sampling_rate, (
            f"Mismatched sampling rate: extractor expects {self.sampling_rate}, "
            f"got {sampling_rate}"
        )
        
        if len(samples.shape) > 1:
            if samples.shape[0] == 2:
                samples = np.mean(samples, axis=0)
            else:
                samples = samples[0]
        
        mel = librosa.feature.melspectrogram(
            y=samples,
            sr=sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=True,
            power=1,
        )
        logmel = np.log(np.clip(mel, a_min=1e-7, a_max=None))
        
        logmel = logmel.T
        
        return logmel

