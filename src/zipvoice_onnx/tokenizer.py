class Tokenizer:
    def __init__(self, token_file: str):
        self.token2id = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = line.rstrip().split("\t")
                if len(parts) == 2:
                    token, token_id = parts[0], int(parts[1])
                    self.token2id[token] = token_id
    
    def phonemes_to_token_ids(self, phonemes: str):
        tokens = list(phonemes.strip())
        
        token_ids = []
        for token in tokens:
            if token in self.token2id:
                token_ids.append(self.token2id[token])
        
        return token_ids
