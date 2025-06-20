import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import ChatDataset
from model.transformer_chat import Seq2SeqTransformer


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChatDataset("data/processed/dialogs.jsonl")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    vocab_size = dataset.tokenizer.vocab_size

    model = Seq2SeqTransformer(vocab_size).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(loader) * 5  # epochs=5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.pad_token_id)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in loader:
            src = batch["input_ids"].T.to(device)
            tgt = batch["labels"].T.to(device)
            tgt_input = tgt[:-1, :]
            src_mask = torch.zeros(src.size(0), src.size(0)).type(torch.bool).to(device)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
            src_pad_mask = (src == dataset.tokenizer.pad_token_id).T.to(device)
            tgt_pad_mask = (tgt_input == dataset.tokenizer.pad_token_id).T.to(device)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "best_model.pt")
    print("✅ 학습 완료, 체크포인트 저장: best_model.pt")


if __name__ == "__main__":
    train_model()