import datetime as dt
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from lhotse.utils import fix_random_seed
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, split_on_silence
from torch import Tensor, nn
from vocos import Vocos

# Hardcoded Hebrew model paths
HEBREW_MODEL_FILES = {
    "text_encoder": "./model_heb/text_encoder.onnx",
    "fm_decoder": "./model_heb/fm_decoder.onnx",
    "text_encoder_int8": "./model_heb/text_encoder_int8.onnx",
    "fm_decoder_int8": "./model_heb/fm_decoder_int8.onnx",
    "model_json": "./model_heb/model.json",
    "tokens": "./model_heb/tokens.txt",
}

# Hardcoded phonemes
HARDCODED_PHONEMES = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."
HARDCODED_PHONEMES_REF = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."

# Hardcoded configuration values
ONNX_INT8 = False
PROMPT_WAV = "prompt.wav"
RES_WAV_PATH = "audio.wav"
GUIDANCE_SCALE = 1.0
NUM_STEP = 16
FEAT_SCALE = 0.1
SPEED = 1.0
T_SHIFT = 0.5
TARGET_RMS = 0.1
SEED = 666
NUM_THREAD = 1
REMOVE_LONG_SIL = False

# Punctuation set for chunking
punctuation = {";", ":", ",", ".", "!", "?", "；", "：", "，", "。", "！", "？"}


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
        samples: Union[np.ndarray, torch.Tensor],
        sampling_rate: int,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract mel spectrogram features from audio samples.

        Args:
            samples: PyTorch tensor or numpy array with shape (C, T) where C is channels, T is time.
            sampling_rate: Sampling rate of the audio.

        Returns:
            PyTorch tensor or numpy array with shape (T, n_mels) containing log mel spectrogram features.
        """
        # Check for sampling rate compatibility
        assert sampling_rate == self.sampling_rate, (
            f"Mismatched sampling rate: extractor expects {self.sampling_rate}, "
            f"got {sampling_rate}"
        )
        
        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

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

        if is_numpy:
            return logmel.cpu().numpy()
        else:
            return logmel


def chunk_tokens_punctuation(tokens_list: List[str], max_tokens: int = 100):
    """
    Splits the input tokens list into chunks according to punctuations,
        each with a maximum number of tokens.

    Args:
        token_list (list of str): The list of tokens to be split.
        max_tokens (int): The maximum number of tokens per chunk.

    Returns:
        List[str]: A list of text chunks.
    """

    # 1. Split the tokens according to punctuations.
    sentences = []
    current_sentence = []
    for token in tokens_list:
        # If the first token of current sentence is punctuation or blank,
        # append it to the end of the previous sentence.
        if (
            len(current_sentence) == 0
            and len(sentences) != 0
            and (token in punctuation or token == " ")
        ):
            sentences[-1].append(token)
        # Otherwise, append the current token to the current sentence.
        else:
            current_sentence.append(token)
            # Split the sentence in positions of punctuations.
            if token in punctuation:
                sentences.append(current_sentence)
                current_sentence = []
    # Assume the last few tokens are also a sentence
    if len(current_sentence) != 0:
        sentences.append(current_sentence)

    # 2. Merge short sentences.
    chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk.extend(sentence)
        else:
            if len(current_chunk) > 0:
                chunks.append(current_chunk)
            current_chunk = sentence

    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    return chunks


def cross_fade_concat(
    chunks: List[torch.Tensor], fade_duration: float = 0.1, sample_rate: int = 24000
) -> torch.Tensor:
    """
    Concatenates audio chunks with cross-fading between consecutive chunks.

    Args:
        chunks: List of audio tensors, each with shape (C, T) where
                C = number of channel, T = time dimension (samples)
        fade_duration: Duration of cross-fade in seconds
        sample_rate: Audio sample rate in Hz

    Returns:
        Concatenated audio tensor with shape (N, T_total)
    """
    # Handle edge cases: empty input or single chunk
    if len(chunks) <= 1:
        return chunks[0] if chunks else torch.tensor([])

    # Calculate total fade samples from duration and sample rate
    fade_samples = int(fade_duration * sample_rate)

    # Use simple concatenation if fade duration is non-positive
    if fade_samples <= 0:
        return torch.cat(chunks, dim=-1)

    # Initialize final tensor with the first chunk
    final = chunks[0]

    # Iterate through remaining chunks to apply cross-fading
    for next_chunk in chunks[1:]:
        # Calculate safe fade length (cannot exceed either chunk's duration)
        k = min(fade_samples, final.shape[-1], next_chunk.shape[-1])

        # Fall back to simple concatenation if safe fade length is invalid
        if k <= 0:
            final = torch.cat([final, next_chunk], dim=-1)
            continue

        # Create fade curve (1 -> 0) with shape (1, k) for broadcasting
        fade = torch.linspace(1, 0, k, device=final.device)[None]

        # Concatenate three parts:
        # 1. Non-overlapping part of previous audio
        # 2. Cross-faded overlapping region
        # 3. Non-overlapping part of next audio
        final = torch.cat(
            [
                final[..., :-k],  # All samples except last k from previous
                final[..., -k:] * fade
                + next_chunk[..., :k] * (1 - fade),  # Cross-fade region
                next_chunk[..., k:],  # All samples except first k from next
            ],
            dim=-1,
        )

    return final


def load_prompt_wav(prompt_wav: str, sampling_rate: int):
    """
    Load the waveform with torchaudio and resampling if needed.

    Parameters:
        prompt_wav: path of the prompt wav.
        sampling_rate: target sampling rate.

    Returns:
        Loaded prompt waveform with target sampling rate,
        PyTorch tensor of shape (C, T)
    """
    prompt_wav, prompt_sampling_rate = torchaudio.load(prompt_wav)

    if prompt_sampling_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=prompt_sampling_rate, new_freq=sampling_rate
        )
        prompt_wav = resampler(prompt_wav)
    return prompt_wav


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
        only_edge: If true, only remove edge silences.
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


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute the intermediate time steps for sampling.

    Args:
        t_start: The starting time of the sampling (default is 0).
        t_end: The starting time of the sampling (default is 1).
        num_step: The number of sampling.
        t_shift: shift the t toward smaller numbers so that the sampling
            will emphasize low SNR region. Should be in the range of (0, 1].
            The shifting will be more significant when the number is smaller.
        device: A torch device.
    Returns:
        The time step with the shape (num_step + 1,).
    """

    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)

    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

    return timesteps


class Tokenizer:
    """Simple tokenizer that reads token mappings from a file and converts text to tokens."""

    def __init__(self, token_file: str):
        """
        Args:
            token_file: Path to token file with format '{token}\t{token_id}' per line.
        """
        self.token2id: dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                if len(info) == 2:
                    token, token_id = info[0], int(info[1])
                    self.token2id[token] = token_id
        self.pad_id = self.token2id.get("_", 0)
        self.vocab_size = len(self.token2id)

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        """
        Convert text strings to lists of character tokens.
        
        Args:
            texts: List of text strings to tokenize.
            
        Returns:
            List of lists of token strings (each character becomes a token).
        """
        return [list(text.strip()) for text in texts]

    def tokens_to_token_ids(self, tokens_list: List[List[str]]) -> List[List[int]]:
        """
        Convert token lists to token ID lists.
        
        Args:
            tokens_list: List of lists of token strings.
            
        Returns:
            List of lists of token IDs.
        """
        token_ids_list = []
        for tokens in tokens_list:
            token_ids = []
            for token in tokens:
                if token in self.token2id:
                    token_ids.append(self.token2id[token])
                else:
                    logging.debug(f"Skip OOV token: {token}")
            token_ids_list.append(token_ids)
        return token_ids_list


class OnnxModel:
    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        num_thread: int = 1,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = num_thread
        session_opts.intra_op_num_threads = num_thread

        self.session_opts = session_opts

        self.init_text_encoder(text_encoder_path)
        self.init_fm_decoder(fm_decoder_path)

    def init_text_encoder(self, model_path: str):
        self.text_encoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_fm_decoder(self, model_path: str):
        self.fm_decoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        out = self.text_encoder.run(
            [
                self.text_encoder.get_outputs()[0].name,
            ],
            {
                self.text_encoder.get_inputs()[0].name: tokens.numpy(),
                self.text_encoder.get_inputs()[1].name: prompt_tokens.numpy(),
                self.text_encoder.get_inputs()[2].name: prompt_features_len.numpy(),
                self.text_encoder.get_inputs()[3].name: speed.numpy(),
            },
        )
        return torch.from_numpy(out[0])

    def run_fm_decoder(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        out = self.fm_decoder.run(
            [
                self.fm_decoder.get_outputs()[0].name,
            ],
            {
                self.fm_decoder.get_inputs()[0].name: t.numpy(),
                self.fm_decoder.get_inputs()[1].name: x.numpy(),
                self.fm_decoder.get_inputs()[2].name: text_condition.numpy(),
                self.fm_decoder.get_inputs()[3].name: speech_condition.numpy(),
                self.fm_decoder.get_inputs()[4].name: guidance_scale.numpy(),
            },
        )
        return torch.from_numpy(out[0])


def sample(
    model: OnnxModel,
    tokens: List[List[int]],
    prompt_tokens: List[List[int]],
    prompt_features: Tensor,
    speed: float = 1.0,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
) -> torch.Tensor:
    """
    Generate acoustic features, given text tokens, prompts feature and prompt
    transcription's text tokens.

    Args:
        tokens: a list of list of text tokens.
        prompt_tokens: a list of list of prompt tokens.
        prompt_features: the prompt feature with the shape
            (batch_size, seq_len, feat_dim).
        speed : speed control.
        t_shift: time shift.
        guidance_scale: the guidance scale for classifier-free guidance.
        num_step: the number of steps to use in the ODE solver.
    """
    # Run text encoder
    assert len(tokens) == len(prompt_tokens) == 1
    tokens = torch.tensor(tokens, dtype=torch.int64)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.int64)
    prompt_features_len = torch.tensor(prompt_features.size(1), dtype=torch.int64)
    speed = torch.tensor(speed, dtype=torch.float32)

    text_condition = model.run_text_encoder(
        tokens, prompt_tokens, prompt_features_len, speed
    )

    batch_size, num_frames, _ = text_condition.shape
    assert batch_size == 1
    feat_dim = model.feat_dim

    # Run flow matching model
    timesteps = get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_step,
        t_shift=t_shift,
    )
    x = torch.randn(batch_size, num_frames, feat_dim)
    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, num_frames - prompt_features.shape[1])
    )  # (B, T, F)
    guidance_scale = torch.tensor(guidance_scale, dtype=torch.float32)

    for step in range(num_step):
        v = model.run_fm_decoder(
            t=timesteps[step],
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale,
        )
        x = x + v * (timesteps[step + 1] - timesteps[step])

    x = x[:, prompt_features_len.item() :, :]
    return x


def generate_sentence(
    save_path: str,
    prompt_wav: str,
    model: OnnxModel,
    vocoder: nn.Module,
    tokenizer: Tokenizer,
    feature_extractor: VocosFbank,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    remove_long_sil: bool = False,
):
    """
    Generate waveform using hardcoded phonemes based on a given prompt waveform.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_wav (str): Path to the prompt wav file.
        model (OnnxModel): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (Tokenizer): The tokenizer used to convert text to tokens.
        feature_extractor: The feature extractor used to
            extract acoustic features.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
        remove_long_sil (bool, optional): Whether to remove long silences in the
            middle of the generated speech (edge silences will be removed by default).
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    # Load and process prompt wav
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)

    # Remove edge and long silences in the prompt wav.
    # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
    prompt_wav = remove_silence(
        prompt_wav, sampling_rate, only_edge=False, trail_sil=200
    )

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    if prompt_duration > 20:
        logging.warning(
            f"Given prompt wav is too long ({prompt_duration}s). "
            f"Please provide a shorter one (1-3 seconds is recommended)."
        )
    elif prompt_duration > 10:
        logging.warning(
            f"Given prompt wav is long ({prompt_duration}s). "
            f"It will lead to slower inference speed and possibly worse speech quality."
        )

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    # Use hardcoded phonemes
    text = HARDCODED_PHONEMES
    prompt_text = HARDCODED_PHONEMES_REF

    # Tokenize text (str tokens), punctuations will be preserved.
    tokens_str = tokenizer.texts_to_tokens([text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]

    # chunk text so that each len(prompt wav + generated wav) is around 25 seconds.
    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (
        len(prompt_tokens_str) * speed
    )
    max_tokens = int((25 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
    print(len(chunked_tokens_str))
    print(chunked_tokens_str)

    # Tokenize text (int tokens)
    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    # Start predicting features
    chunked_features = []
    start_t = dt.datetime.now()
    for tokens in chunked_tokens:

        # Generate features
        pred_features = sample(
            model=model,
            tokens=[tokens],
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            speed=speed,
            t_shift=t_shift,
            guidance_scale=guidance_scale,
            num_step=num_step,
        )

        # Postprocess predicted features
        pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)
        chunked_features.append(pred_features)

    # Start vocoder processing
    chunked_wavs = []
    start_vocoder_t = dt.datetime.now()

    for pred_features in chunked_features:
        wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
        # Adjust wav volume if necessary
        if prompt_rms < target_rms:
            wav = wav * prompt_rms / target_rms
        chunked_wavs.append(wav)

    # Finish model generation
    t = (dt.datetime.now() - start_t).total_seconds()

    # Merge chunked wavs
    final_wav = cross_fade_concat(
        chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate
    )
    final_wav = remove_silence(
        final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0
    )

    # Calculate processing time metrics
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)
    return metrics


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

@torch.inference_mode()
def main():
    torch.set_num_threads(NUM_THREAD)
    torch.set_num_interop_threads(NUM_THREAD)

    fix_random_seed(SEED)

    # Use hardcoded Hebrew model paths
    if ONNX_INT8:
        text_encoder_name = "text_encoder_int8"
        fm_decoder_name = "fm_decoder_int8"
    else:
        text_encoder_name = "text_encoder"
        fm_decoder_name = "fm_decoder"

    text_encoder_path = Path(HEBREW_MODEL_FILES[text_encoder_name])
    fm_decoder_path = Path(HEBREW_MODEL_FILES[fm_decoder_name])
    model_config_path = Path(HEBREW_MODEL_FILES["model_json"])
    token_file = Path(HEBREW_MODEL_FILES["tokens"])

    if not text_encoder_path.is_file():
        raise FileNotFoundError(f"{text_encoder_path} does not exist")
    if not fm_decoder_path.is_file():
        raise FileNotFoundError(f"{fm_decoder_path} does not exist")
    if not model_config_path.is_file():
        raise FileNotFoundError(f"{model_config_path} does not exist")
    if not token_file.is_file():
        raise FileNotFoundError(f"{token_file} does not exist")

    logging.info(f"Using Hebrew model from {text_encoder_path.parent}")

    # Initialize tokenizer
    tokenizer = Tokenizer(token_file=token_file)

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    model = OnnxModel(text_encoder_path, fm_decoder_path, num_thread=NUM_THREAD)

    # Initialize vocoder and feature extractor
    vocoder = get_vocoder(None)
    vocoder.eval()

    if model_config["feature"]["type"] == "vocos":
        feature_extractor = VocosFbank()
    else:
        raise NotImplementedError(
            f"Unsupported feature type: {model_config['feature']['type']}"
        )
    sampling_rate = model_config["feature"]["sampling_rate"]

    logging.info("Start generating...")
    generate_sentence(
        save_path=RES_WAV_PATH,
        prompt_wav=PROMPT_WAV,
        model=model,
        vocoder=vocoder,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        num_step=NUM_STEP,
        guidance_scale=GUIDANCE_SCALE,
        speed=SPEED,
        t_shift=T_SHIFT,
        target_rms=TARGET_RMS,
        feat_scale=FEAT_SCALE,
        sampling_rate=sampling_rate,
        remove_long_sil=REMOVE_LONG_SIL,
    )
    logging.info(f"Saved to: {RES_WAV_PATH}")
    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    main()

