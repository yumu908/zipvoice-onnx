# -*- coding: utf-8 -*-
import os
import asyncio
import sys
import json
import logging
import io
import base64
import re
import zipfile
import shutil
import tempfile
import urllib.request
import numpy as np
import soundfile as sf
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize
import pypinyin

from zipvoice_onnx import ZipVoice, ZipVoiceOptions
from zipvoice_onnx.audio import load_prompt_wav, remove_silence, cross_fade_concat
from zipvoice_onnx.vocoder import rms_norm
from zipvoice_onnx.model import sample
from zipvoice_onnx.tokenizer import chunk_tokens_punctuation

# Ensure logger config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zipvoice_server")

# Default mapping for built-in prompt voices to their transcripts (phonemes/text)
DEFAULT_REF_TEXTS = {
    "prompt_english_female1.wav": "In order to win, you must expect to win.",
    "prompt_english_female2.wav": "In order to win, you must expect to win.",
    "prompt.wav": "In order to win, you must expect to win.",
}

# 中文标点映射到英文半角标点
PUNCTUATION_MAP = {
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "；": ";",
    "：": ":",
    "、": ",",
}

# Global ZipVoice instance and valid tokens set
zipvoice: Optional[ZipVoice] = None
valid_tokens = set()
zipformer_recognizer = None  # Global cache for Sherpa-ONNX ZipFormer ASR recognizer

def _ensure_sensevoice_model() -> str:
    """Download Alibaba FunASR SenseVoice Small model (SOTA Chinese & English ASR)."""
    model_dir = os.path.abspath("./model/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
    model_file = os.path.join(model_dir, "model.int8.onnx")
    tokens_file = os.path.join(model_dir, "tokens.txt")

    if os.path.exists(model_file) and os.path.exists(tokens_file):
        return model_dir

    os.makedirs(model_dir, exist_ok=True)

    import tarfile as _tarfile

    tarball_url = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
    )
    logger.info(f"Downloading SenseVoice model archive from {tarball_url}...")
    req = urllib.request.Request(tarball_url, headers={"User-Agent": "Mozilla/5.0"})

    with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as tmp:
        tmp_path = tmp.name
        with urllib.request.urlopen(req, timeout=120) as resp:
            shutil.copyfileobj(resp, tmp)

    logger.info("Extracting SenseVoice model files...")
    with _tarfile.open(tmp_path, "r:bz2") as tar:
        for member in tar.getmembers():
            fname = os.path.basename(member.name)
            if fname in ["model.int8.onnx", "tokens.txt"]:
                member.name = fname
                tar.extract(member, model_dir)
    os.unlink(tmp_path)
    logger.info("SenseVoice ASR model ready.")
    return model_dir


def transcribe_audio(audio_path: str) -> str:
    global zipformer_recognizer
    if zipformer_recognizer is None:
        import sysconfig
        sp = sysconfig.get_path("purelib")
        onnx_capi = os.path.join(sp, "onnxruntime", "capi")
        if os.path.isdir(onnx_capi):
            try:
                os.add_dll_directory(onnx_capi)
            except Exception:
                pass
        import sherpa_onnx

        model_dir = _ensure_sensevoice_model()
        model_file = os.path.join(model_dir, "model.int8.onnx")
        tokens_file = os.path.join(model_dir, "tokens.txt")

        logger.info("Initializing Sherpa-ONNX SenseVoice ASR model (Chinese/English SOTA)...")
        zipformer_recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_file,
            tokens=tokens_file,
            num_threads=2,
            use_itn=True,
            provider="cpu",
        )
        logger.info("Sherpa-ONNX SenseVoice model initialized successfully.")
    
    import soundfile as sf
    import librosa
    
    # ZipFormer memory usage scales with audio length; cap at 15s to prevent OOM.
    MAX_ASR_SECONDS = 15
    MAX_ASR_SAMPLES = MAX_ASR_SECONDS * 16000  # at 16kHz after resampling

    logger.info(f"Loading reference audio for transcription: {audio_path}")
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert to mono
        
    if sr != 16000:
        logger.info(f"Resampling reference audio from {sr}Hz to 16000Hz for ZipFormer...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Truncate to prevent excessive memory allocation
    if len(audio) > MAX_ASR_SAMPLES:
        logger.info(f"Audio ({len(audio)/16000:.1f}s) exceeds {MAX_ASR_SECONDS}s limit — truncating for ASR.")
        audio = audio[:MAX_ASR_SAMPLES]
        
    logger.info("Transcribing reference audio using Sherpa-ONNX ZipFormer...")
    stream = zipformer_recognizer.create_stream()
    stream.accept_waveform(16000, audio.astype(np.float32))
    zipformer_recognizer.decode_stream(stream)
    text = stream.result.text.strip()
    logger.info(f"ZipFormer ASR transcription result: '{text}'")
    return text

def _ensure_espeak_data():
    loader_dir = os.path.dirname(espeakng_loader.__file__)
    data_path = os.path.join(loader_dir, 'espeak-ng-data')
    if os.path.isdir(data_path) and os.listdir(data_path):
        return
    logger.info(f"espeak-ng data not found at {data_path}, downloading...")
    url = "https://github.com/espeak-ng/espeak-ng-data/archive/refs/heads/master.zip"
    zip_path, _ = urllib.request.urlretrieve(url)
    
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    extracted_dir = os.path.join(temp_dir, "espeak-ng-data-master")
    
    # Ensure parent dir exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    shutil.move(extracted_dir, data_path)
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("espeak-ng data download complete.")

def load_valid_tokens(tokens_file):
    tokens = set()
    with open(tokens_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                tokens.add(parts[0])
    return tokens

DIGIT_MAP = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}

def num_to_zh(text: str) -> str:
    """Convert digits to Chinese characters so TTS can pronounce them."""
    return "".join(DIGIT_MAP.get(ch, ch) for ch in text)

def _single_zh_segment_to_tokens(text: str, valid_tokens: set) -> list:
    pinyins = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
    initials_list = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]
    
    tokens = []
    for item in pinyins:
        pinyin_str = item[0]
        if len(pinyin_str) == 1 and pinyin_str in valid_tokens:
            tokens.append(pinyin_str)
            continue
            
        if not pinyin_str.isalnum():
            for char in pinyin_str:
                if char in valid_tokens:
                    tokens.append(char)
            continue
            
        tone = ""
        if pinyin_str[-1].isdigit():
            tone = pinyin_str[-1]
            pinyin_base = pinyin_str[:-1]
        else:
            tone = "5"
            pinyin_base = pinyin_str
            
        matched_initial = ""
        for init in initials_list:
            if pinyin_base.startswith(init):
                matched_initial = init
                break
                
        if matched_initial:
            initial_token = matched_initial + "0"
            final_base = pinyin_base[len(matched_initial):]
            
            if matched_initial == "y" and final_base == "u":
                final_base = "v"
            elif matched_initial in ["j", "q", "x"] and final_base == "u":
                final_base = "v"
            elif matched_initial in ["n", "l"] and final_base == "v":
                final_base = "v"
            elif matched_initial in ["n", "l"] and final_base == "ve":
                final_base = "ve"
                
            final_token = final_base + tone
            tokens.append(initial_token)
            tokens.append(final_token)
        else:
            final_token = pinyin_base + tone
            tokens.append(final_token)
            
    return tokens

def chinese_to_zipvoice_tokens(text: str, valid_tokens: set) -> list:
    """Convert mixed Chinese, English, and numbers into valid ZipVoice tokens."""
    # 1. Replace Chinese punctuation with half-width punctuation
    for ch_punc, en_punc in PUNCTUATION_MAP.items():
        text = text.replace(ch_punc, en_punc)
        
    # 2. Convert digits to Chinese characters (e.g. '2026' -> '二零二六')
    text = num_to_zh(text)
    
    # 3. Segment by English words vs Chinese/punctuation
    parts = re.split(r'([a-zA-Z]+)', text)
    
    tokens = []
    for part in parts:
        if not part:
            continue
        if re.match(r'^[a-zA-Z]+$', part):
            # English word in Chinese text: convert to espeak phonemes
            try:
                phonemes = phonemize(part, language="en-us", backend="espeak").strip()
                for char in phonemes:
                    if char in valid_tokens:
                        tokens.append(char)
            except Exception:
                # Fallback to single letters
                for char in part.lower():
                    if char in valid_tokens:
                        tokens.append(char)
        else:
            # Chinese segment or punctuation
            sub_tokens = _single_zh_segment_to_tokens(part, valid_tokens)
            tokens.extend(sub_tokens)
            
    return tokens

def custom_texts_to_tokens(texts):
    res = []
    for text in texts:
        parts = text.strip().split()
        is_space_separated_tokens = any(
            len(p) > 1 and (p[-1].isdigit() or p.endswith('0'))
            for p in parts
        )
        if is_space_separated_tokens:
            res.append(parts)
        else:
            res.append(list(text.strip()))
    return res

def text_to_phonemes(text: str, language: str) -> str:
    if language == "zh":
        tokens = chinese_to_zipvoice_tokens(text, valid_tokens)
        return " ".join(tokens)
    else:
        phonemes = phonemize(text=text, language="en-us", backend="espeak")
        return phonemes

def split_text_into_sentences(text: str, language: str) -> List[str]:
    if language == "zh":
        sentences = re.split(r'([。！；？\n])', text)
        chunks = []
        for i in range(0, len(sentences) - 1, 2):
            chunk = sentences[i] + sentences[i+1]
            chunks.append(chunk)
        if len(sentences) % 2 == 1 and sentences[-1]:
            chunks.append(sentences[-1])
        return [c.strip() for c in chunks if c.strip()]
    else:
        # Split on standard English sentence boundaries
        sentences = re.split(r'([.!?\n])', text)
        chunks = []
        for i in range(0, len(sentences) - 1, 2):
            chunk = sentences[i] + sentences[i+1]
            chunks.append(chunk)
        if len(sentences) % 2 == 1 and sentences[-1]:
            chunks.append(sentences[-1])
        return [c.strip() for c in chunks if c.strip()]

def resolve_path(relative_path: str) -> str:
    """Resolve a path relative to the executable directory if frozen, otherwise relative to the current working directory."""
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        path = os.path.abspath(os.path.join(base_dir, relative_path))
        if os.path.exists(path):
            return path
    return os.path.abspath(relative_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global zipvoice, valid_tokens
    
    # 1. Setup espeak-ng data
    try:
        _ensure_espeak_data()
        os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        logger.info("Espeak-ng loaded successfully.")
    except Exception as e:
        logger.error(f"Error ensuring espeak-ng data: {e}")
        
    # 2. Setup model directories (prefer unified 'model' folder)
    model_dir = resolve_path("model")
    if not os.path.exists(model_dir):
        model_dir = resolve_path("model-en-distilled")
    
    # Check if CUDA is available, if not, use INT8 models for CPU speedup
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    logger.info(f"Available ONNX providers: {available_providers}")
    
    use_int8 = "CUDAExecutionProvider" not in available_providers
    
    text_encoder_name = "text_encoder_int8.onnx" if use_int8 else "text_encoder.onnx"
    fm_decoder_name = "fm_decoder_int8.onnx" if use_int8 else "fm_decoder.onnx"
    
    text_encoder_path = os.path.join(model_dir, text_encoder_name)
    fm_decoder_path = os.path.join(model_dir, fm_decoder_name)
    
    if not os.path.exists(text_encoder_path):
        text_encoder_path = os.path.join(model_dir, "text_encoder.onnx")
        fm_decoder_path = os.path.join(model_dir, "fm_decoder.onnx")
        logger.warning("Target model path not found, falling back to FP32 models.")
    else:
        if use_int8:
            logger.info("CUDA not found. Automatically loaded INT8 quantized models for faster CPU inference.")
        else:
            logger.info("CUDA found. Loaded FP32 models for GPU inference.")
        
    options = ZipVoiceOptions(
        text_encoder_path=text_encoder_path,
        fm_decoder_path=fm_decoder_path,
        text_encoder_int8_path=os.path.join(model_dir, "text_encoder_int8.onnx"),
        fm_decoder_int8_path=os.path.join(model_dir, "fm_decoder_int8.onnx"),
        model_json_path=os.path.join(model_dir, "model.json"),
        tokens_path=os.path.join(model_dir, "tokens.txt"),
        vocoder_path=os.path.join(model_dir, "vocos_24khz.onnx"),
        onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    
    logger.info(f"Initializing ZipVoice engine using {model_dir} ...")
    zipvoice = ZipVoice(options)
    # Patch texts_to_tokens with custom parser for space-separated & custom tokens
    zipvoice.tokenizer.texts_to_tokens = custom_texts_to_tokens
    
    valid_tokens = load_valid_tokens(options.tokens_path)
    logger.info("ZipVoice engine startup successfully completed.")
    yield

app = FastAPI(
    title="ZipVoice ONNX API",
    description="A FastAPI wrapper around ZipVoice ONNX model supporting non-streaming & streaming HTTP, and WebSockets.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SynthesizeRequest(BaseModel):
    text: str
    language: str = "zh"  # 'zh' or 'en'
    ref_wav: str = "examples/audio/prompt_english_female1.wav"
    ref_text: Optional[str] = None
    speed: float = 1.0
    num_steps: int = 4
    guidance_scale: float = 1.0
    t_shift: float = 0.5

class TranscribeRequest(BaseModel):
    path: str

def resolve_ref_wav(ref_wav: str) -> str:
    if not ref_wav:
        return ""
    if os.path.exists(ref_wav) and os.path.isfile(ref_wav):
        return ref_wav
    base_name = os.path.basename(ref_wav)
    # Check in examples/audio
    examples_path = os.path.join("./examples/audio", base_name)
    if os.path.exists(examples_path) and os.path.isfile(examples_path):
        return examples_path
    # Check in uploads
    upload_path = os.path.join("./uploads", base_name)
    if os.path.exists(upload_path) and os.path.isfile(upload_path):
        return upload_path
    return ref_wav

@app.get("/api/voices")
def get_voices():
    """List available reference voice files from examples/audio and uploads."""
    voices = []
    # Predefined examples/audio voices
    examples_audio_dir = "./examples/audio"
    if os.path.exists(examples_audio_dir):
        for f in os.listdir(examples_audio_dir):
            if f.endswith(".wav") and os.path.isfile(os.path.join(examples_audio_dir, f)):
                voices.append({"name": f, "path": os.path.join(examples_audio_dir, f), "type": "system"})
            
    # Uploaded voices
    upload_dir = "./uploads"
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            if f.endswith(".wav") and os.path.isfile(os.path.join(upload_dir, f)):
                voices.append({"name": f, "path": os.path.join(upload_dir, f), "type": "uploaded"})
    return voices

@app.post("/api/upload")
async def upload_voice(file: UploadFile = File(...)):
    """Upload a custom reference audio (.wav only)."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
        
    upload_dir = "./uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    return {"name": file.filename, "path": file_path, "type": "uploaded"}

@app.get("/api/voices/audio")
def get_voice_audio(path: str):
    """Retrieve the raw audio file for previewing."""
    resolved = resolve_ref_wav(path)
    if resolved and os.path.exists(resolved) and os.path.isfile(resolved):
        return FileResponse(resolved, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found")

@app.post("/api/voices/transcribe")
async def transcribe_voice(req: TranscribeRequest):
    """Transcribe a reference audio file using Whisper."""
    resolved = resolve_ref_wav(req.path)
    if not resolved or not os.path.exists(resolved) or not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail=f"Reference wav {req.path} not found")
            
    try:
        text = transcribe_audio(resolved)
        return {"text": text}
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Generate whole audio and return as a WAV file (Non-streaming)."""
    global zipvoice
    if not zipvoice:
        raise HTTPException(status_code=503, detail="ZipVoice engine is not initialized yet.")
        
    ref_wav = resolve_ref_wav(req.ref_wav)
    # Validate reference file existence
    if not ref_wav or not os.path.exists(ref_wav) or not os.path.isfile(ref_wav):
        raise HTTPException(status_code=404, detail=f"Reference wav {req.ref_wav} not found")
            
    ref_text = req.ref_text
    if not ref_text:
        ref_text = DEFAULT_REF_TEXTS.get(os.path.basename(ref_wav))
        if not ref_text:
            try:
                ref_text = transcribe_audio(ref_wav)
            except Exception as e:
                logger.error(f"Whisper auto-transcription failed: {e}")
                ref_text = "In order to win, you must expect to win."
        
    try:
        ref_lang = "zh" if any('\u4e00' <= char <= '\u9fff' for char in ref_text) else "en"
        ref_phonemes = text_to_phonemes(ref_text, ref_lang)
        target_phonemes = text_to_phonemes(req.text, req.language)
        
        logger.info(f"Synthesizing non-streaming: {req.text[:20]}... using {ref_wav}")
        samples, sample_rate = zipvoice.create(
            ref_wav=ref_wav,
            ref_phonemes=ref_phonemes,
            target_phonemes=target_phonemes,
            speed=req.speed,
            num_steps=req.num_steps,
            guidance_scale=req.guidance_scale,
            t_shift=req.t_shift
        )
        
        out_buf = io.BytesIO()
        sf.write(out_buf, samples, sample_rate, format='WAV')
        out_buf.seek(0)
        
        return StreamingResponse(out_buf, media_type="audio/wav", headers={
            "Content-Disposition": f"attachment; filename=synthesized.wav"
        })
    except Exception as e:
        logger.error(f"Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def generate_chunks(req: SynthesizeRequest, ref_wav: str, ref_text: str):
    global zipvoice, valid_tokens
    
    target_rms = 0.1
    feat_scale = 0.1
    
    # 1. Feature Pre-Extraction (Same logic as chinese_loop.py)
    prompt_wav = load_prompt_wav(ref_wav, sampling_rate=zipvoice.sampling_rate)
    prompt_wav = remove_silence(prompt_wav, zipvoice.sampling_rate, only_edge=False, trail_sil=200)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
    prompt_duration = prompt_wav.shape[-1] / zipvoice.sampling_rate

    prompt_features = zipvoice.feature_extractor.extract(prompt_wav, sampling_rate=zipvoice.sampling_rate)
    prompt_features = prompt_features[np.newaxis] * feat_scale  # (1, T, n_mels)

    ref_lang = "zh" if any('\u4e00' <= char <= '\u9fff' for char in ref_text) else "en"
    ref_phonemes = text_to_phonemes(ref_text, ref_lang)
    prompt_tokens_str = zipvoice.tokenizer.texts_to_tokens([ref_phonemes])[0]
    prompt_tokens = zipvoice.tokenizer.tokens_to_token_ids([prompt_tokens_str])
    
    # 2. Split sentences
    sentences = split_text_into_sentences(req.text, req.language)
    
    for idx, sentence in enumerate(sentences):
        logger.info(f"Streaming sentence {idx + 1}/{len(sentences)}: {sentence[:20]}...")
        if req.language == "zh":
            target_tokens = chinese_to_zipvoice_tokens(sentence, valid_tokens)
        else:
            target_tokens = zipvoice.tokenizer.texts_to_tokens([text_to_phonemes(sentence, "en")])[0]
            
        token_duration = prompt_duration / (len(prompt_tokens_str) * req.speed) if len(prompt_tokens_str) > 0 else 0.15
        calc_max = int((25 - prompt_duration) / token_duration) if token_duration > 0 else 50
        max_tokens = max(10, min(calc_max, 100))
        
        chunked_tokens_str = chunk_tokens_punctuation(target_tokens, max_tokens=max_tokens)
        chunked_tokens = zipvoice.tokenizer.tokens_to_token_ids(chunked_tokens_str)
        
        chunk_wavs = []
        for tokens in chunked_tokens:
            pred_features = sample(
                model=zipvoice.model,
                tokens=[tokens],
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                speed=req.speed,
                t_shift=req.t_shift,
                guidance_scale=req.guidance_scale,
                num_step=req.num_steps,
            )
            pred_features = (np.transpose(pred_features, (0, 2, 1)) / feat_scale).astype(np.float32)
            wav = zipvoice.vocoder.decode(pred_features)
            wav = np.clip(wav, -1, 1)
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            chunk_wavs.append(wav)
            
        # Combine
        sentence_wav = cross_fade_concat(chunk_wavs, fade_duration=0.1, sample_rate=zipvoice.sampling_rate)
        sentence_wav = remove_silence(sentence_wav, zipvoice.sampling_rate, only_edge=True, trail_sil=0)
        
        if sentence_wav.shape[0] == 1:
            sentence_wav = sentence_wav[0]
            
        # Transform to 16-bit PCM bytes
        pcm_bytes = (sentence_wav * 32767).astype(np.int16).tobytes()
        
        # Base64 encode
        b64_audio = base64.b64encode(pcm_bytes).decode("utf-8")
        
        yield {
            "index": idx,
            "text": sentence,
            "audio": b64_audio,
            "sample_rate": zipvoice.sampling_rate,
            "done": idx == len(sentences) - 1
        }

@app.post("/api/generate_stream")
async def synthesize_stream(req: SynthesizeRequest, format: str = "json"):
    """
    Generate audio sentence-by-sentence and stream.
    - format="json": yields lines of line-delimited JSON with base64 audio and metadata.
    - format="pcm": yields raw binary PCM bytes (16-bit, 24kHz, mono).
    """
    global zipvoice
    if not zipvoice:
        raise HTTPException(status_code=503, detail="ZipVoice engine is not initialized yet.")
        
    ref_wav = resolve_ref_wav(req.ref_wav)
    if not ref_wav or not os.path.exists(ref_wav) or not os.path.isfile(ref_wav):
        raise HTTPException(status_code=404, detail=f"Reference wav {req.ref_wav} not found")
            
    ref_text = req.ref_text or DEFAULT_REF_TEXTS.get(os.path.basename(ref_wav))
    if not ref_text:
        try:
            ref_text = transcribe_audio(ref_wav)
        except Exception as e:
            logger.error(f"Whisper auto-transcription failed: {e}")
            ref_text = "In order to win, you must expect to win."
    
    if format == "json":
        async def json_generator():
            try:
                async for chunk in generate_chunks(req, ref_wav, ref_text):
                    yield json.dumps(chunk) + "\n"
            except Exception as e:
                logger.error(f"Error in stream generator: {e}")
                yield json.dumps({"error": str(e)}) + "\n"
        return StreamingResponse(json_generator(), media_type="application/x-ndjson")
    else:
        async def pcm_generator():
            try:
                async for chunk in generate_chunks(req, ref_wav, ref_text):
                    yield base64.b64decode(chunk["audio"])
            except Exception as e:
                logger.error(f"Error in pcm generator: {e}")
        return StreamingResponse(pcm_generator(), media_type="audio/l16; rate=24000; channels=1")

@app.websocket("/api/synthesize")
async def ws_synthesize(websocket: WebSocket):
    """
    WebSocket endpoint for speech synthesis with real-time interruption.
    Supports persistent connection and barge-in (interruption).
    """
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    current_task = None
    
    async def run_synthesis(req_params):
        try:
            req = SynthesizeRequest(**req_params)
            ref_wav = resolve_ref_wav(req.ref_wav)
            if not ref_wav or not os.path.exists(ref_wav) or not os.path.isfile(ref_wav):
                await websocket.send_json({"event": "error", "message": f"Reference wav {req.ref_wav} not found"})
                return
            
            ref_text = req.ref_text or DEFAULT_REF_TEXTS.get(os.path.basename(ref_wav))
            if not ref_text:
                try:
                    ref_text = transcribe_audio(ref_wav)
                except Exception as e:
                    logger.error(f"Whisper auto-transcription failed: {e}")
                    ref_text = "In order to win, you must expect to win."
            
            async for chunk in generate_chunks(req, ref_wav, ref_text):
                await websocket.send_json({
                    "event": "audio",
                    "index": chunk["index"],
                    "text": chunk["text"],
                    "audio": chunk["audio"],
                    "sample_rate": chunk["sample_rate"],
                    "done": chunk["done"]
                })
                await asyncio.sleep(0)  # Yield control to event loop to allow interruption processing
            await websocket.send_json({"event": "done"})
            logger.info("WebSocket synthesis done successfully.")
        except asyncio.CancelledError:
            logger.info("Synthesis task was interrupted/cancelled.")
            try:
                await websocket.send_json({"event": "interrupted"})
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"Error during WS synthesis: {e}")
            try:
                await websocket.send_json({"event": "error", "message": str(e)})
            except Exception:
                pass

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            event = msg.get("event", "synthesize")
            
            # Cancel the active task if one is running
            if current_task and not current_task.done():
                current_task.cancel()
                try:
                    await current_task
                except asyncio.CancelledError:
                    pass
            
            if event == "synthesize":
                params = msg.get("params", msg)
                current_task = asyncio.create_task(run_synthesis(params))
            elif event == "interrupt":
                logger.info("Client requested interruption.")
                # We already cancelled the task above, so we just acknowledge the interruption
                await websocket.send_json({"event": "interrupted"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection disconnected by client.")
    except Exception as e:
        logger.error(f"Error inside WebSocket main loop: {e}")
    finally:
        if current_task and not current_task.done():
            current_task.cancel()
        try:
            await websocket.close()
        except:
            pass

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    fav_path = os.path.join(os.path.dirname(__file__), "web", "favicon.ico")
    if os.path.exists(fav_path):
        return FileResponse(fav_path, media_type="image/x-icon")
    icon_root = resolve_path("icon.ico")
    if os.path.exists(icon_root):
        return FileResponse(icon_root, media_type="image/x-icon")
    raise HTTPException(status_code=404, detail="Favicon not found")

# Serve static web files
web_dir = os.path.join(os.path.dirname(__file__), "web")
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
else:
    @app.get("/")
    def index():
        return HTMLResponse("<h3>Web files not found. Place web files in src/zipvoice_onnx/web/</h3>")
