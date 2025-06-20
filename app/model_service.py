import torch
from transformers import PreTrainedTokenizerFast
from model.transformer_chat import Seq2SeqTransformer


class ChatService:
    def __init__(self, ckpt_path="../best_model.pt", tokenizer_name="bert-base-multilingual-cased", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
        self.model = Seq2SeqTransformer(self.tokenizer.vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    def generate(self, text: str, max_len=64) -> str:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_len).input_ids.T.to(self.device)
        mask = torch.zeros(tokens.size(0), tokens.size(0), dtype=torch.bool).to(self.device)
        out = self.model(tokens, tokens, mask, mask, None, None)
        pred_ids = out.argmax(-1).T[0].tolist()
        return self.tokenizer.decode(pred_ids, skip_special_tokens=True)