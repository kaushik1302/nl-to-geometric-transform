"""Turn every instruction into numbers with four text representations.

Writes data/features/<split>_<representation>.npy: one row per instruction,
in the same order as data/splits/<split>.jsonl.
"""
import json
import os
import re

import numpy as np
import open_clip
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from dyooptix.data import vocab

SPLITS = ["train", "val", "test", "test_unseen_templates", "geobench"]

# Every meaningful phrase in the training vocabulary (object words are left out)
KEYWORDS = (vocab.TRANSLATE_VERBS + vocab.ROTATE_VERBS
            + vocab.SCALE_UP_VERBS + vocab.SCALE_DOWN_VERBS
            + list(vocab.TRANSLATE_DIRECTIONS)
            + [p for phrases in vocab.ROTATE_DIRECTIONS.values() for p in phrases]
            + [w for words in vocab.ADVERBS.values() for w in words])


def hand_features(sentence):
    """A 0/1 flag per keyword, then the three explicit numbers (0 when absent)."""
    padded = " " + sentence.replace(",", " ") + " "
    flags = [int(" " + k + " " in padded) for k in KEYWORDS]

    px = re.search(r"by (\d+) pixels", sentence)
    deg = re.search(r"by (\d+) degrees", sentence)
    fac = re.search(r"by a factor of ([\d.]+)", sentence)
    numbers = [int(px.group(1)) / 512 if px else 0,
               int(deg.group(1)) if deg else 0,
               float(np.log(float(fac.group(1)))) if fac else 0]
    return flags + numbers


def clip_vectors(sentences, model, tokenizer):
    """CLIP text vectors (512 numbers each), scaled to length 1, in batches of 256."""
    chunks = []
    with torch.no_grad():
        for i in range(0, len(sentences), 256):
            v = model.encode_text(tokenizer(sentences[i:i + 256]))
            chunks.append(v / v.norm(dim=-1, keepdim=True))
    return torch.cat(chunks).numpy()


if __name__ == "__main__":
    os.makedirs("data/features", exist_ok=True)

    sentences = {}
    for split in SPLITS:
        with open(f"data/splits/{split}.jsonl", encoding="utf-8") as f:
            sentences[split] = [json.loads(line)["instruction"] for line in f]

    tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit(sentences["train"])
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    minilm = SentenceTransformer("all-MiniLM-L6-v2")

    for split, sents in sentences.items():
        features = {
            "hand": np.array([hand_features(s) for s in sents]),
            "tfidf": tfidf.transform(sents).toarray(), # type: ignore
            "clip": clip_vectors(sents, clip_model, clip_tokenizer),
            "minilm": minilm.encode(sents, batch_size=256),
        }
        for name, x in features.items():
            np.save(f"data/features/{split}_{name}.npy", x.astype(np.float32))
            print(f"{split:22} {name:7} {x.shape}")