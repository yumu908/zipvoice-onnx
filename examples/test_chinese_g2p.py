import pypinyin

def load_valid_tokens(tokens_file):
    valid_tokens = set()
    with open(tokens_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                valid_tokens.add(parts[0])
    return valid_tokens

def chinese_to_zipvoice_tokens(text, valid_tokens):
    pinyins = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
    initials_list = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]
    
    tokens = []
    for item in pinyins:
        pinyin_str = item[0]
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
            
            # Special mappings:
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

# Test
valid = load_valid_tokens("model-en-distilled/tokens.txt")
test_text = "你好，世界！这是一只特别可爱的猫咪。绿色，雨天，决定，去哪儿。"
tokens = chinese_to_zipvoice_tokens(test_text, valid)
print("Original text:", test_text)
print("Generated tokens:", tokens)

invalid_tokens = [t for t in tokens if t not in valid]
if invalid_tokens:
    print("Invalid tokens (not in vocabulary):", invalid_tokens)
else:
    print("All generated tokens are valid!")
