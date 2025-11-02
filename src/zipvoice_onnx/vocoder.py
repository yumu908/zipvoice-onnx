from typing import Optional

import numpy as np
import torch
import torchaudio
from torch import nn
from vocos import Vocos


class VocosFbank:
    """Feature extractor for computing mel spectrograms compatible with Vocos vocoder."""

    def __init__(
        self,
        sampling_rate: int = 24000,
        n_mels: int = 100,
        n_fft: int = 1024,
        hop_length: int = 256,
        num_channels: int = 1,
    ):
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_channels = num_channels
        
        # Use torchaudio transforms to match training setup
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )

    def extract(
        self,
        samples: torch.Tensor,
        sampling_rate: int,
    ) -> torch.Tensor:
        """
        Extract mel spectrogram features from audio samples.

        Args:
            samples: PyTorch tensor with shape (C, T) where C is channels, T is time.
            sampling_rate: Sampling rate of the audio.

        Returns:
            PyTorch tensor with shape (T, n_mels) containing log mel spectrogram features.
        """
        # Check for sampling rate compatibility
        assert sampling_rate == self.sampling_rate, (
            f"Mismatched sampling rate: extractor expects {self.sampling_rate}, "
            f"got {sampling_rate}"
        )
        
        # Ensure samples have the right shape: (C, T)
        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        
        # Handle multi-channel audio
        if self.num_channels == 1:
            if samples.shape[0] == 2:
                samples = samples.mean(dim=0, keepdims=True)
        else:
            assert samples.shape[0] == 2, samples.shape

        # Compute mel spectrogram using torchaudio
        mel = self.fbank(samples)  # (1, n_mels, T) or (2, n_mels, T)
        logmel = mel.clamp(min=1e-7).log()

        # Reshape to (T, n_mels) or (T, 2 * n_mels)
        logmel = logmel.reshape(-1, logmel.shape[-1]).t()  # (time, n_mels) or (time, 2 * n_mels)

        return logmel


def get_vocoder(vocos_local_path: Optional[str] = None):
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder


def rms_norm(prompt_wav: torch.Tensor, target_rms: float):
    """
    Normalize the rms of prompt_wav is it is smaller than target rms.

    Parameters:
        prompt_wav: PyTorch tensor with shape (C, T).
        target_rms: target rms value

    Returns:
        prompt_wav: normalized prompt wav with shape (C, T).
        promt_rms: rms of original prompt wav. Will be used to
            re-normalize the generated wav.
    """
    prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
    if prompt_rms < target_rms:
        prompt_wav = prompt_wav * target_rms / prompt_rms
    return prompt_wav, prompt_rms

