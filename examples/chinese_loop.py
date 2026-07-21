# -*- coding: utf-8 -*-
"""
优化版：运行长文本/循环生成中文文本的示例。
利用特征复用（Pre-computed Voice Prompt Features）避免重复的文件读取和 Mel 提取。
使用方式：
python examples/chinese_loop.py
"""

import os
import re
import numpy as np
import soundfile as sf
import pypinyin
from zipvoice_onnx import ZipVoice, ZipVoiceOptions

# 导入 zipvoice-onnx 的内部推理及处理函数
from zipvoice_onnx.model import sample
from zipvoice_onnx.tokenizer import chunk_tokens_punctuation
from zipvoice_onnx.audio import load_prompt_wav, remove_silence, cross_fade_concat
from zipvoice_onnx.vocoder import rms_norm

# 中文标点映射到英文标点
PUNCTUATION_MAP = {
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "；": ";",
    "：": ":",
    "、": ",",
}

def load_valid_tokens(tokens_file):
    valid_tokens = set()
    with open(tokens_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                valid_tokens.add(parts[0])
    return valid_tokens

def chinese_to_zipvoice_tokens(text, valid_tokens):
    for ch_punc, en_punc in PUNCTUATION_MAP.items():
        text = text.replace(ch_punc, en_punc)
        
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

def split_into_sentences(text):
    # 按照常用断句标点切分
    sentences = re.split(r'([。！；？\n])', text)
    chunks = []
    for i in range(0, len(sentences) - 1, 2):
        chunk = sentences[i] + sentences[i+1]
        chunks.append(chunk)
    if len(sentences) % 2 == 1 and sentences[-1]:
        chunks.append(sentences[-1])
    # 过滤空句，并去除多余的首尾空格/换行
    return [c.strip() for c in chunks if c.strip()]

def main():
    options = ZipVoiceOptions(
        text_encoder_path="./model-en-distilled/text_encoder.onnx",
        fm_decoder_path="./model-en-distilled/fm_decoder.onnx",
        text_encoder_int8_path="./model-en-distilled/text_encoder_int8.onnx",
        fm_decoder_int8_path="./model-en-distilled/fm_decoder_int8.onnx",
        model_json_path="./model-en-distilled/model.json",
        tokens_path="./model-en-distilled/tokens.txt",
        vocoder_path="./vocos_24khz.onnx",
        # 优先使用 GPU (CUDA) 加速，若不支持则自动回退至 CPU
        onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    
    zipvoice = ZipVoice(options)
    zipvoice.tokenizer.texts_to_tokens = custom_texts_to_tokens
    
    # ----------------- 1. 一次性加载并提取参考音频特征 -----------------
    ref_wav = "examples/audio/prompt_english_female1.wav"
    ref_phonemes = "ɪn ˈɔɹdəɹ tə wˈɪn, ju mˈʌst ɪkspˈɛkt tə wˈɪn."
    target_rms = 0.1
    feat_scale = 0.1
    
    print("正在提取参考音频特征...")
    prompt_wav = load_prompt_wav(ref_wav, sampling_rate=zipvoice.sampling_rate)
    prompt_wav = remove_silence(prompt_wav, zipvoice.sampling_rate, only_edge=False, trail_sil=200)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
    prompt_duration = prompt_wav.shape[-1] / zipvoice.sampling_rate

    prompt_features = zipvoice.feature_extractor.extract(prompt_wav, sampling_rate=zipvoice.sampling_rate)
    prompt_features = prompt_features[np.newaxis] * feat_scale  # (1, T, n_mels)

    prompt_tokens_str = zipvoice.tokenizer.texts_to_tokens([ref_phonemes])[0]
    prompt_tokens = zipvoice.tokenizer.tokens_to_token_ids([prompt_tokens_str])
    print("参考特征提取完成。")
    
    # ----------------- 2. 切分长文本 -----------------
    long_text = (
        "大漠飞雪，是一场跨越时空的凄美邂逅。金黄浩瀚的沙海，本是烈日与焦土的领地，却在寒流骤起之时，被琼屑改写了疆界。"
        "狂风卷着雪花，在大地上奔走呼号，千万朵洁白在沙丘间碰撞、破碎、又旋转而起。"
        "沙子是暖色的苍凉，雪花是冷色的孤傲，二者在交织中褪去了原本的颜色，幻化成一种混沌而深邃的苍茫。"
        "这一幕，既是苍穹的馈赠，也是荒野的叹息。行人踏过，脚印瞬间被风雪填平，留不住过往，更看不清前路。"
    )
    
    sentences = split_into_sentences(long_text)
    print(f"\n待合成长文本共有 {len(sentences)} 句，已开始循环合成...")
    
    # ----------------- 3. 循环高效合成 -----------------
    valid_tokens = load_valid_tokens(options.tokens_path)
    all_chunk_wavs = []
    
    speed = 1.0
    num_steps = 4
    t_shift = 0.5
    guidance_scale = 1.0
    
    for idx, sentence in enumerate(sentences, 1):
        # 转换为拼音 Token 列表
        target_tokens = chinese_to_zipvoice_tokens(sentence, valid_tokens)
        
        # 计算 token 的限制以分段（避开 ONNX OOM）
        token_duration = prompt_duration / (len(prompt_tokens_str) * speed)
        max_tokens = int((25 - prompt_duration) / token_duration)
        max_tokens = min(max_tokens, 1000)
        
        chunked_tokens_str = chunk_tokens_punctuation(target_tokens, max_tokens=max_tokens)
        chunked_tokens = zipvoice.tokenizer.tokens_to_token_ids(chunked_tokens_str)
        
        # 循环推理当前句子的各个 Token 切片
        for tokens in chunked_tokens:
            pred_features = sample(
                model=zipvoice.model,
                tokens=[tokens],
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                speed=speed,
                t_shift=t_shift,
                guidance_scale=guidance_scale,
                num_step=num_steps,
            )
            pred_features = (np.transpose(pred_features, (0, 2, 1)) / feat_scale).astype(np.float32)
            wav = zipvoice.vocoder.decode(pred_features)
            wav = np.clip(wav, -1, 1)
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            all_chunk_wavs.append(wav)
            
        print(f"-> 已完成第 {idx}/{len(sentences)} 句: '{sentence[:10]}...'")
        
    # ----------------- 4. 拼接及淡入淡出处理 -----------------
    print("开始拼接所有音频片段...")
    final_wav = cross_fade_concat(all_chunk_wavs, fade_duration=0.1, sample_rate=zipvoice.sampling_rate)
    final_wav = remove_silence(final_wav, zipvoice.sampling_rate, only_edge=True, trail_sil=0)
    
    if final_wav.shape[0] == 1:
        final_wav = final_wav[0]
        
    # 保存合成结果
    out_path = "audio_chinese_long.wav"
    sf.write(out_path, final_wav, zipvoice.sampling_rate)
    print(f"【成功】长文本音频已保存至: {out_path}，总时长 {final_wav.shape[0]/zipvoice.sampling_rate:.2f} 秒。")

if __name__ == "__main__":
    main()
