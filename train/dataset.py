import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class ChatDataset(Dataset):
    def __init__(self, path, tokenizer_name="bert-base-multilingual-cased", max_len=64):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.pairs = [json.loads(l) for l in open(path, encoding="utf-8")]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        src = self.tokenizer(pair["q"], truncation=True, padding="max_length",
                             max_length=self.max_len, return_tensors="pt")
        tgt = self.tokenizer(pair["a"], truncation=True, padding="max_length",
                             max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": src.input_ids.squeeze(0),
            "attention_mask": src.attention_mask.squeeze(0),
            "labels": tgt.input_ids.squeeze(0)
        }