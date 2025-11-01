from dataclasses import dataclass
from typing import Tuple
import json
import numpy as np

from .model import OnnxModel
from .vocoder import get_vocoder, VocosFbank
from .tokenizer import Tokenizer
from .utils import load_prompt_wav, rms_norm, sample


@dataclass
class ZipVoiceOptions:
    text_encoder_path: str
    fm_decoder_path: str
    text_encoder_int8_path: str = ""
    fm_decoder_int8_path: str = ""
    model_json_path: str = ""
    tokens_path: str = ""
    vocos_model_path: str = ""


class ZipVoice:
    def __init__(self, options: ZipVoiceOptions):
        self.options = options
        
        use_int8 = bool(options.text_encoder_int8_path and options.fm_decoder_int8_path)
        text_encoder_path = options.text_encoder_int8_path if use_int8 else options.text_encoder_path
        fm_decoder_path = options.fm_decoder_int8_path if use_int8 else options.fm_decoder_path
        
        self.model = OnnxModel(text_encoder_path, fm_decoder_path, num_thread=1)
        
        self.model_config = None
        if options.model_json_path:
            with open(options.model_json_path, "r") as f:
                self.model_config = json.load(f)
            self.sampling_rate = self.model_config["feature"]["sampling_rate"]
        else:
            self.sampling_rate = 24000
        
        if options.tokens_path:
            self.tokenizer = Tokenizer(options.tokens_path)
        else:
            self.tokenizer = None
        
        vocoder_path = options.vocos_model_path if options.vocos_model_path else None
        self.vocoder = get_vocoder(vocoder_path)
        self.vocoder.eval()
        
        self.feature_extractor = VocosFbank(sampling_rate=self.sampling_rate)
        
        self.is_distill = "distill" in text_encoder_path.lower() or "distill" in options.model_json_path.lower()
    
    def create(
        self,
        ref_wav: str,
        ref_phonemes: str,
        target_phonemes: str,
        speed: float = 1.0,
        num_steps: int | None = None,
    ) -> Tuple[np.ndarray, int]:
        if self.is_distill:
            default_num_step = 8
            guidance_scale = 3.0
        else:
            default_num_step = 16
            guidance_scale = 1.0
        
        num_step = num_steps if num_steps is not None else default_num_step
        
        t_shift = 0.5
        target_rms = 0.1
        feat_scale = 0.1
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Please provide tokens_path in ZipVoiceOptions.")
        
        tokens = self.tokenizer.phonemes_to_token_ids(target_phonemes)
        prompt_tokens = self.tokenizer.phonemes_to_token_ids(ref_phonemes)
        
        prompt_wav_tensor = load_prompt_wav(ref_wav, sampling_rate=self.sampling_rate)
        prompt_wav_tensor, prompt_rms = rms_norm(prompt_wav_tensor, target_rms)
        
        prompt_features = self.feature_extractor.extract(prompt_wav_tensor[0], sampling_rate=self.sampling_rate)
        prompt_features = np.expand_dims(prompt_features, axis=0).astype(np.float32) * feat_scale
        
        pred_features = sample(
            model=self.model,
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            speed=speed,
            t_shift=t_shift,
            guidance_scale=guidance_scale,
            num_step=num_step,
        )
        
        pred_features = np.transpose(pred_features, (0, 2, 1)) / feat_scale
        
        wav = self.vocoder.decode(pred_features)
        wav = np.squeeze(wav, axis=1)
        wav = np.clip(wav, -1, 1)
        
        if prompt_rms < target_rms:
            wav = wav * prompt_rms / target_rms
        
        samples = wav[0]
        return samples, self.sampling_rate
