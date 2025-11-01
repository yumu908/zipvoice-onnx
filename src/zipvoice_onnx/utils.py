import numpy as np
import soundfile as sf
import librosa
from .model import OnnxModel


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
) -> np.ndarray:
    timesteps = np.linspace(t_start, t_end, num_step + 1, dtype=np.float32)
    timesteps = (t_shift * timesteps / (1 + (t_shift - 1) * timesteps)).astype(np.float32)
    return timesteps


def load_prompt_wav(prompt_wav: str, sampling_rate: int):
    data, prompt_sampling_rate = sf.read(prompt_wav)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data[np.newaxis, :]
    
    if prompt_sampling_rate != sampling_rate:
        data = librosa.resample(data[0], orig_sr=prompt_sampling_rate, target_sr=sampling_rate)
        data = data[np.newaxis, :]
    return data


def rms_norm(prompt_wav: np.ndarray, target_rms: float):
    prompt_rms = np.sqrt(np.mean(np.square(prompt_wav)))
    if prompt_rms < target_rms:
        prompt_wav = prompt_wav * target_rms / prompt_rms
    return prompt_wav, prompt_rms


def sample(model: OnnxModel, tokens: list, prompt_tokens: list, 
          prompt_features: np.ndarray, speed: float = 1.0, t_shift: float = 0.5,
          guidance_scale: float = 1.0, num_step: int = 16) -> np.ndarray:
    tokens = np.array([tokens], dtype=np.int64)
    prompt_tokens = np.array([prompt_tokens], dtype=np.int64)
    prompt_features_len = np.array([prompt_features.shape[1]], dtype=np.int64)
    speed_tensor = np.array([speed], dtype=np.float32)

    text_condition = model.run_text_encoder(
        tokens, prompt_tokens, prompt_features_len, speed_tensor
    )

    batch_size, num_frames, _ = text_condition.shape
    feat_dim = model.feat_dim

    timesteps = get_time_steps(
        t_start=0.0, t_end=1.0, num_step=num_step, t_shift=t_shift
    )
    np.random.seed(666)
    x = np.random.randn(batch_size, num_frames, feat_dim).astype(np.float32)
    
    pad_width = ((0, 0), (0, num_frames - prompt_features.shape[1]), (0, 0))
    speech_condition = np.pad(prompt_features, pad_width, mode='constant').astype(np.float32)
    
    guidance_scale_tensor = np.array(guidance_scale, dtype=np.float32)

    for step in range(num_step):
        t_step = np.array(timesteps[step], dtype=np.float32)
        dt_step = np.float32(timesteps[step + 1] - timesteps[step])
        v = model.run_fm_decoder(
            t=t_step,
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale_tensor,
        )
        x = (x + v * dt_step).astype(np.float32)

    x = x[:, prompt_features_len[0]:, :]
    return x

