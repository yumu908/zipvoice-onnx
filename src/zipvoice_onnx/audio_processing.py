import numpy as np
import torch
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, split_on_silence


def audiosegment_to_tensor(aseg):
    """
    Convert a pydub.AudioSegment to PyTorch audio tensor
    """
    audio_data = np.array(aseg.get_array_of_samples())

    # Convert to float32 and normalize to [-1, 1] range
    audio_data = audio_data.astype(np.float32) / 32768.0

    # Handle channels
    if aseg.channels == 1:
        # Mono channel: add channel dimension (T) -> (1, T)
        tensor_data = torch.from_numpy(audio_data).unsqueeze(0)
    else:
        # Multi-channel: reshape to (C, T)
        tensor_data = torch.from_numpy(audio_data.reshape(-1, aseg.channels).T)

    return tensor_data


def tensor_to_audiosegment(tensor, sample_rate):
    """
    Convert a PyTorch audio tensor to pydub.AudioSegment

    Parameters:
        tensor: Tensor with shape (C, T), where C is the number of channels
            and T is the time steps
        sample_rate: Audio sample rate
    """
    # Convert tensor to numpy array
    audio_np = tensor.cpu().numpy()

    # Add channel dimension if single channel
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]

    # Convert to int16 type (common format for pydub)
    # Assumes tensor values are in [-1, 1] range as floating point
    audio_np = (audio_np * 32768.0).clip(-32768, 32767).astype(np.int16)

    # Convert to byte stream
    # For multi-channel audio, pydub requires interleaved format
    # (e.g., left-right-left-right)
    if audio_np.shape[0] > 1:
        # Convert to interleaved format
        audio_np = audio_np.transpose(1, 0).flatten()
    audio_bytes = audio_np.tobytes()

    # Create AudioSegment
    audio_segment = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=tensor.shape[0],
    )

    return audio_segment


def remove_silence_edges(
    audio: AudioSegment, keep_silence: int = 100, silence_threshold: float = -50
):
    """
    Remove edge silences longer than `keep_silence` ms.

    Parameters:
        audio: an AudioSegment object.
        keep_silence: kept silence in the edge.
        silence_threshold: the threshold of silence.

    Returns:
        An AudioSegment object
    """
    # Remove leading silence
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - keep_silence)
    audio = audio[start_idx:]

    # Remove trailing silence
    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - keep_silence)
    audio = audio[start_idx:]
    audio = audio.reverse()

    return audio


def remove_silence(
    audio: torch.Tensor,
    sampling_rate: int,
    only_edge: bool = False,
    trail_sil: float = 0,
):
    """
    Remove silences longer than 1 second, and edge silences longer than 0.1 seconds

    Parameters:
        audio: PyTorch tensor with shape (C, T).
        sampling_rate: sampling rate of the audio.
        only_edge: If true, only remove edge silences.
        trail_sil: the duration of added trailing silence in ms.

    Returns:
        PyTorch tensor with shape (C, T), where C is number of channels
            and T is number of audio samples
    """
    # Load audio file
    wave = tensor_to_audiosegment(audio, sampling_rate)

    if not only_edge:
        # Split audio using silences longer than 1 second
        non_silent_segs = split_on_silence(
            wave,
            min_silence_len=1000,  # Silences longer than 1 second (1000ms)
            silence_thresh=-50,
            keep_silence=1000,  # Keep 1.0 second of silence around segments
            seek_step=10,
        )

        # Concatenate all non-silent segments
        wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            wave += seg

    # Remove silence longer than 0.1 seconds in the begining and ending of wave
    wave = remove_silence_edges(wave, 100, -50)

    # Add trailing silence to avoid leaking prompt to generated speech.
    wave = wave + AudioSegment.silent(duration=trail_sil)

    # Convert to PyTorch tensor
    return audiosegment_to_tensor(wave)

