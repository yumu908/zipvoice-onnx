# -*- coding: utf-8 -*-
"""
超长文本（2万字）循环生成中文文本的高效示例。
1. 使用 tqdm 显示合并与合成进度。
2. 特征一键复用，单次提取参考音频，避开重复 IO。
3. 按照标点切句，防止长文本在 ONNX 推理时 OOM 并提高合成自然度。
4. 使用显卡 GPU (CUDA) 进行并行加速推理。
"""

import os
import re
import numpy as np
import soundfile as sf
import pypinyin
from tqdm import tqdm
from zipvoice_onnx import ZipVoice, ZipVoiceOptions

# 导入 zipvoice-onnx 的核心处理和解码器
from zipvoice_onnx.model import sample
from zipvoice_onnx.tokenizer import chunk_tokens_punctuation
from zipvoice_onnx.audio import load_prompt_wav, remove_silence, cross_fade_concat
from zipvoice_onnx.vocoder import rms_norm

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
    # 根据句末标点和换行进行切分
    sentences = re.split(r'([。！；？\n])', text)
    chunks = []
    for i in range(0, len(sentences) - 1, 2):
        chunk = sentences[i] + sentences[i+1]
        chunks.append(chunk)
    if len(sentences) % 2 == 1 and sentences[-1]:
        chunks.append(sentences[-1])
    return [c.strip() for c in chunks if c.strip()]

def generate_large_text():
    # 模拟一个长文本段落
    base_paragraph = (
        "大漠飞雪，是一场跨越时空的凄美邂逅。金黄浩瀚的沙海，本是烈日与焦土的领地，却在寒流骤起之时，被漫天席卷而来的琼屑改写了疆界。"
        "狂风卷着雪花，在大地上奔走呼号，千万朵洁白在沙丘间碰撞、破碎、又旋转而起。"
        "沙子是暖色的苍凉，雪花是冷色的孤傲，二者在交织中褪去了原本的颜色，幻化成一种混沌而深邃的苍茫。"
        "那些平日里狰狞的沙脊，此刻竟显出几分江南水墨般的清逸，仿佛亘古的黄沙在这一刻，披上了一件圣洁而厚重的素衣。"
        "这一幕，既是苍穹的馈赠，也是荒野的叹息。行人踏过，脚印瞬间被风雪填平，留不住过往，更看不清前路。"
        "此时的大漠，不再是单调的沉寂，而是被天地间极致的对比所震撼，于粗犷中生出几分极致的柔情。"
        "这不仅仅是气候的奇景，更是一场关于纯粹与辽阔的诗意修行，令人在凛冽中感受到生命最深处的震撼。\n"
    )
    # 复制 40 次以达到约 2 万字（约 20,000 字符）
    long_text = base_paragraph * 40
    return long_text

def main():
    options = ZipVoiceOptions(
        text_encoder_path="./model-en-distilled/text_encoder.onnx",
        fm_decoder_path="./model-en-distilled/fm_decoder.onnx",
        text_encoder_int8_path="./model-en-distilled/text_encoder_int8.onnx",
        fm_decoder_int8_path="./model-en-distilled/fm_decoder_int8.onnx",
        model_json_path="./model-en-distilled/model.json",
        tokens_path="./model-en-distilled/tokens.txt",
        vocoder_path="./vocos_24khz.onnx",
        onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    
    zipvoice = ZipVoice(options)
    zipvoice.tokenizer.texts_to_tokens = custom_texts_to_tokens
    
    # 1. 加载和预提取参考音频的特征
    ref_wav = "examples/audio/prompt_english_female1.wav"
    ref_phonemes = "ɪn ˈɔɹdəɹ tə wˈɪn, ju mˈʌst ɪkspˈɛkt tə wˈɪn."
    target_rms = 0.1
    feat_scale = 0.1
    
    print(">>> 正在提取声音克隆源(参考音频)的特征...")
    prompt_wav = load_prompt_wav(ref_wav, sampling_rate=zipvoice.sampling_rate)
    prompt_wav = remove_silence(prompt_wav, zipvoice.sampling_rate, only_edge=False, trail_sil=200)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
    prompt_duration = prompt_wav.shape[-1] / zipvoice.sampling_rate

    prompt_features = zipvoice.feature_extractor.extract(prompt_wav, sampling_rate=zipvoice.sampling_rate)
    prompt_features = prompt_features[np.newaxis] * feat_scale  # (1, T, n_mels)

    prompt_tokens_str = zipvoice.tokenizer.texts_to_tokens([ref_phonemes])[0]
    prompt_tokens = zipvoice.tokenizer.tokens_to_token_ids([prompt_tokens_str])
    print(">>> 参考音频特征提取成功。")
    
    # 2. 生成或读取超长文本
    # 如果您有本地文本文件，可以取消注释并修改为：
    # with open("your_story.txt", "r", encoding="utf-8") as f:
    #     target_text = f.read()
    target_text = generate_large_text()
    
    total_len = len(target_text)
    sentences = split_into_sentences(target_text)
    print(f"\n>>> 待合成的总文本字数：约 {total_len} 字。")
    print(f">>> 文本已自动按照标点切分为 {len(sentences)} 个句子块。")
    
    # 3. 循环高效推理（复用参考音频的特征）
    valid_tokens = load_valid_tokens(options.tokens_path)
    all_chunk_wavs = []
    
    speed = 1.0
    num_steps = 4
    t_shift = 0.5
    guidance_scale = 1.0
    
    # 使用 tqdm 进度条显示句子的合成进度
    for idx, sentence in enumerate(tqdm(sentences, desc="正在合成句子区块", unit="句")):
        target_tokens = chinese_to_zipvoice_tokens(sentence, valid_tokens)
        
        # 计算切片边界
        token_duration = prompt_duration / (len(prompt_tokens_str) * speed)
        max_tokens = int((25 - prompt_duration) / token_duration)
        max_tokens = min(max_tokens, 1000)
        
        chunked_tokens_str = chunk_tokens_punctuation(target_tokens, max_tokens=max_tokens)
        chunked_tokens = zipvoice.tokenizer.tokens_to_token_ids(chunked_tokens_str)
        
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
            
    # 4. 音频最后大合并
    print("\n>>> 正在拼接音频块并做淡入淡出平滑处理...")
    final_wav = cross_fade_concat(all_chunk_wavs, fade_duration=0.1, sample_rate=zipvoice.sampling_rate)
    final_wav = remove_silence(final_wav, zipvoice.sampling_rate, only_edge=True, trail_sil=0)
    
    if final_wav.shape[0] == 1:
        final_wav = final_wav[0]
        
    out_path = "audio_chinese_20k.wav"
    sf.write(out_path, final_wav, zipvoice.sampling_rate)
    
    duration_min = (final_wav.shape[0] / zipvoice.sampling_rate) / 60
    print(f"\n【成功】全部超长文本合成完毕！")
    print(f">>> 结果音频保存至: {out_path}")
    print(f">>> 生成音频总时长约: {duration_min:.2f} 分钟。")

if __name__ == "__main__":
    main()
