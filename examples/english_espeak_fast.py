# -*- coding: utf-8 -*-
"""
english_espeak_fast.py

一个轻量级示例，展示如何用 ZipVoice‑ONNX 高效合成较长的中文/英文文本。

特性
----
* 使用仓库自带的 **int8（量化）模型**，CPU 上约快 2‑3 倍。
* 将长文本切分为约 400 字的块，分别 phonemize、推理，再把生成的音频块拼接。
* 在当前机器上仅使用 **CPUExecutionProvider**（原先的 DmlExecutionProvider 不可用）。
* 自动下载缺失的 espeak‑ng 数据文件，并设置 `ESPEAK_DATA_PATH`，防止 `phontab` 找不到的错误。

运行方式
--------
cd d:\\englishApp\\zipvoice-onnx
python examples\\english_espeak_fast.py

脚本将在项目根目录生成 ``audio_fast.wav`` 并在控制台打印每块的生成时长。
"""

import os
import urllib.request
import zipfile
import tempfile
import shutil
import numpy as np
import soundfile as sf
import re

from zipvoice_onnx import ZipVoice, ZipVoiceOptions
from phonemizer import phonemize
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Helper: download missing espeak‑ng data (idempotent)
# ---------------------------------------------------------------------------
def _ensure_espeak_data():
    data_path = espeakng_loader.get_data_path()
    if os.path.isdir(data_path) and os.listdir(data_path):
        # already present
        return
    print(f"espeak‑ng data not found at {data_path}, downloading…")
    url = "https://github.com/espeak-ng/espeak-ng-data/archive/refs/heads/master.zip"
    zip_path, _ = urllib.request.urlretrieve(url)
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    extracted_dir = os.path.join(temp_dir, "espeak-ng-data-master")
    # copy the whole directory to the expected location
    shutil.copytree(extracted_dir, data_path, dirs_exist_ok=True)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("espeak‑ng data download complete.")
    # make sure the library can find the data
    os.environ["ESPEAK_DATA_PATH"] = data_path


# ---------------------------------------------------------------------------
# Helper: split a long string into roughly `max_chars` characters.
# Tries to break on punctuation first for nicer prosody.
# ---------------------------------------------------------------------------
def split_into_chunks(text: str, max_chars: int = 200) -> list:
    """Split the input text into fixed-size chunks of up to `max_chars` characters.
    This deterministic chunking avoids large intermediate tensors that can cause OOM.
    """
    # Ensure we handle empty strings gracefully
    if not text:
        return []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Ensure espeak‑ng data is available
    _ensure_espeak_data()
    # 2. Point Espeak to the correct library / data path
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

    # -----------------------------------------------------------------------
    # Model configuration – use the int8 (quantised) models for speed
    # -----------------------------------------------------------------------
    model_dir = "./model"
    options = ZipVoiceOptions(
        text_encoder_path=os.path.join(model_dir, "text_encoder_int8.onnx"),
        fm_decoder_path=os.path.join(model_dir, "fm_decoder_int8.onnx"),
        text_encoder_int8_path=os.path.join(model_dir, "text_encoder_int8.onnx"),
        fm_decoder_int8_path=os.path.join(model_dir, "fm_decoder_int8.onnx"),
        model_json_path=os.path.join(model_dir, "model.json"),
        tokens_path=os.path.join(model_dir, "tokens.txt"),
        vocoder_path="./vocos_24khz.onnx",
        # Only CPU is guaranteed to be present on this machine
        onnx_providers=["CPUExecutionProvider"],
    )

    zipvoice = ZipVoice(options)

    # -----------------------------------------------------------------------
    # Reference audio & text (speaker)
    # -----------------------------------------------------------------------
    ref_wav = "prompt_english_female1.wav"
    ref_text = "In order to win, you must expect to win."
    ref_phonemes = phonemize(text=ref_text, language="en-us", backend="espeak")

    # -----------------------------------------------------------------------
    # Target (long) text – you can replace this with any long paragraph
    # -----------------------------------------------------------------------
    target_text = "There is a sublime, terrifying silence beneath the gale; the shifting sands are quickly buried under a pristine, shifting mantle of frost. The endless ridges of the desert, usually defined by their sweeping curves, are now softened and blurred by the blinding curtain of snow. It is a collision of extremes—the biting cold of the tundra meeting the infinite isolation of the wasteland. Amidst this tempest, the desert feels both ancient and ephemeral, a vast kingdom reclaimed by the fury of the winter wind, where time itself seems to freeze in the heart of the storm."

    # -----------------------------------------------------------------------
    # Split, synthesize, and concatenate
    # -----------------------------------------------------------------------
    chunks = split_into_chunks(target_text, max_chars=50)
    print(f"目标文本被切分为 {len(chunks)} 块")

    all_samples = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_phonemes = phonemize(text=chunk, language="en-us", backend="espeak")
        samples, sr = zipvoice.create(ref_wav, ref_phonemes, chunk_phonemes)
        print(f"[Chunk {idx}/{len(chunks)}] 生成时长 {samples.shape[0] / sr:.2f}s")
        all_samples.append(samples)

    final_audio = np.concatenate(all_samples)
    out_path = "audio_fast.wav"
    sf.write(out_path, final_audio, sr)
    print(f"已保存合成音频至 {out_path}，总时长 {final_audio.shape[0] / sr:.2f}s")
