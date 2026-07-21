# -*- coding: utf-8 -*-
"""
运行中文文本合成的示例代码。
使用方式：
python examples/chinese.py
"""

import os
import soundfile as sf
import pypinyin
from zipvoice_onnx import ZipVoice, ZipVoiceOptions

# 中文标点映射到英文标点（因为 tokens.txt 里面只有英文半角标点）
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
    # 替换中文标点
    for ch_punc, en_punc in PUNCTUATION_MAP.items():
        text = text.replace(ch_punc, en_punc)
        
    pinyins = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
    initials_list = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]
    
    tokens = []
    for item in pinyins:
        pinyin_str = item[0]
        # 判断是否为单个有效标点字符
        if len(pinyin_str) == 1 and pinyin_str in valid_tokens:
            tokens.append(pinyin_str)
            continue
            
        if not pinyin_str.isalnum():
            # 其他非字母数字的字符，过滤出存在于词表里的字符
            for char in pinyin_str:
                if char in valid_tokens:
                    tokens.append(char)
            continue
            
        # 提取声调 (1-5)
        tone = ""
        if pinyin_str[-1].isdigit():
            tone = pinyin_str[-1]
            pinyin_base = pinyin_str[:-1]
        else:
            tone = "5"
            pinyin_base = pinyin_str
            
        # 匹配声母
        matched_initial = ""
        for init in initials_list:
            if pinyin_base.startswith(init):
                matched_initial = init
                break
                
        if matched_initial:
            initial_token = matched_initial + "0"
            final_base = pinyin_base[len(matched_initial):]
            
            # 处理 ü 的特殊映射规则
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
            # 零声母的情况，直接为 韵母+声调
            final_token = pinyin_base + tone
            tokens.append(final_token)
            
    return tokens

def custom_texts_to_tokens(texts):
    res = []
    for text in texts:
        parts = text.strip().split()
        # 检查是否包含带声调数字或'0'结尾的拼音 token
        is_space_separated_tokens = any(
            len(p) > 1 and (p[-1].isdigit() or p.endswith('0'))
            for p in parts
        )
        if is_space_separated_tokens:
            res.append(parts)
        else:
            res.append(list(text.strip()))
    return res

def main():
    # 使用 distilled 版本的模型
    options = ZipVoiceOptions(
        text_encoder_path="./model-en-distilled/text_encoder.onnx",
        fm_decoder_path="./model-en-distilled/fm_decoder.onnx",
        text_encoder_int8_path="./model-en-distilled/text_encoder_int8.onnx",
        fm_decoder_int8_path="./model-en-distilled/fm_decoder_int8.onnx",
        model_json_path="./model-en-distilled/model.json",
        tokens_path="./model-en-distilled/tokens.txt",
        vocoder_path="./vocos_24khz.onnx",
        # onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    
    # 实例化 ZipVoice 引擎
    zipvoice = ZipVoice(options)
    
    # 动态把 tokenizer.texts_to_tokens 函数重载为我们自定义的解析器
    zipvoice.tokenizer.texts_to_tokens = custom_texts_to_tokens
    
    # 准备参考音频和其对应的英文音标 (Cross-lingual Voice Cloning)
    ref_wav = "examples/audio/prompt_english_female1.wav"
    ref_phonemes = "ɪn ˈɔɹdəɹ tə wˈɪn, ju mˈʌst ɪkspˈɛkt tə wˈɪn."
    
    # 准备目标中文文本
    target_text = "大漠飞雪，是一场跨越时空的凄美邂逅。金黄浩瀚的沙海，本是烈日与焦土的领地，却在寒流骤起之时，也是荒野的叹息。"
    print(f"目标文本: {target_text}")
    
    # 获取有效的 vocabulary
    valid_tokens = load_valid_tokens(options.tokens_path)
    
    # 转换中文为拼音 Token 列表
    target_tokens = chinese_to_zipvoice_tokens(target_text, valid_tokens)
    target_phonemes = " ".join(target_tokens)
    
    print(f"转换后的拼音 Tokens (前 20 个): {target_tokens[:20]}")
    
    # 调用合成函数
    samples, sample_rate = zipvoice.create(ref_wav, ref_phonemes, target_phonemes, num_steps=4, speed=1.0)
    print(f"成功生成音频: {samples.shape} 采样点，采样率 {sample_rate} Hz")
    
    # 保存音频
    out_path = "audio_chinese.wav"
    sf.write(out_path, samples, sample_rate)
    print(f"音频已保存至: {out_path}")

if __name__ == "__main__":
    main()
