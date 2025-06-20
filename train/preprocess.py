import json, re
from pathlib import Path


def clean_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^가-힣a-z0-9?.!,]+", " ", s)
    return s


def build_jsonl(raw_dir="data/raw", out_path="data/processed/dialogs.jsonl"):
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pairs = []
    for file in Path(raw_dir).glob("*.txt"):
        for line in open(file, encoding="utf-8"):
            if "\t" in line:
                q, a = line.split("\t")[:2]
                pairs.append({"q": clean_text(q), "a": clean_text(a)})
    with open(out_path, "w", encoding="utf-8") as fw:
        for pair in pairs:
            fw.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build_jsonl()
    print("✅ 데이터 전처리 완료: data/processed/dialogs.jsonl")