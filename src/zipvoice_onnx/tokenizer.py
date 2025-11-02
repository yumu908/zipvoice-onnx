import logging
from typing import List


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

