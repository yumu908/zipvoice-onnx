import numpy as np
import soundfile as sf
import librosa


def load_prompt_wav(prompt_wav: str, sampling_rate: int):
    data, prompt_sampling_rate = sf.read(prompt_wav)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data[np.newaxis, :]
    
    if prompt_sampling_rate != sampling_rate:
        data = librosa.resample(data[0], orig_sr=prompt_sampling_rate, target_sr=sampling_rate)
        data = data[np.newaxis, :]
    return data

