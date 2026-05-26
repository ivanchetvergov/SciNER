"""Span-based NER dataset (alternative to BIO token classification).

Each sample carries span candidates over main-sentence words only:
  span_starts, span_ends : token indices of first subword of each word
  span_labels            : 0=background, 1..6=entity type
  span_mask              : False for padding spans
"""
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data import (
    ENTITY_TYPES,
    _CTX,
    load_split_docs,
    build_context_sentences,
    augment_entity_swap,
)

MAX_SPAN_LEN    = 8
NUM_SPAN_CLASSES = len(ENTITY_TYPES) + 1  # 0=background, 1..6


def _etype_id(etype: str) -> int:
    return ENTITY_TYPES.index(etype) + 1


def tokenize_for_spans(
    sentences: list[list[tuple[str, str]]],
    tokenizer,
    max_length: int = 512,
) -> list[dict]:
    """
    Tokenize sentences and enumerate candidate spans over main-sentence words.
    Context-window tokens (_CTX) are excluded from span enumeration.
    """
    results = []
    for sent in sentences:
        tokens      = [w for w, _ in sent]
        word_labels = [l for _, l in sent]

        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        # First-subword positions and BIO labels for main-sentence words only
        word_starts: list[int] = []
        bio_labels:  list[str] = []
        prev_word_id = None
        for tok_idx, word_id in enumerate(enc.word_ids()):
            if word_id is not None and word_id != prev_word_id:
                lbl = word_labels[word_id]
                if lbl != _CTX:
                    word_starts.append(tok_idx)
                    bio_labels.append(lbl)
            prev_word_id = word_id

        # Extract gold entity spans → {(tok_s, tok_e): span_label}
        gold: dict[tuple[int, int], int] = {}
        i = 0
        while i < len(bio_labels):
            lbl = bio_labels[i]
            if lbl.startswith("B-"):
                etype = lbl[2:]
                j = i + 1
                while j < len(bio_labels) and bio_labels[j] == f"I-{etype}":
                    j += 1
                gold[(word_starts[i], word_starts[j - 1])] = _etype_id(etype)
                i = j
            else:
                i += 1

        # Enumerate candidate spans up to MAX_SPAN_LEN words
        span_starts, span_ends, span_labels = [], [], []
        n = len(word_starts)
        for i in range(n):
            for j in range(i, min(i + MAX_SPAN_LEN, n)):
                s, e = word_starts[i], word_starts[j]
                span_starts.append(s)
                span_ends.append(e)
                span_labels.append(gold.get((s, e), 0))

        results.append({
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "span_starts":    span_starts,
            "span_ends":      span_ends,
            "span_labels":    span_labels,
        })

    return results


class SpanNERDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d = self.data[idx]
        return {
            "input_ids":      torch.tensor(d["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(d["attention_mask"], dtype=torch.long),
            "span_starts":    torch.tensor(d["span_starts"],    dtype=torch.long),
            "span_ends":      torch.tensor(d["span_ends"],      dtype=torch.long),
            "span_labels":    torch.tensor(d["span_labels"],    dtype=torch.long),
        }


def span_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids      = pad_sequence([b["input_ids"]      for b in batch], batch_first=True)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True)

    max_spans = max(b["span_starts"].size(0) for b in batch)
    B = len(batch)
    span_starts = torch.zeros(B, max_spans, dtype=torch.long)
    span_ends   = torch.zeros(B, max_spans, dtype=torch.long)
    span_labels = torch.zeros(B, max_spans, dtype=torch.long)
    span_mask   = torch.zeros(B, max_spans, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["span_starts"].size(0)
        span_starts[i, :n] = b["span_starts"]
        span_ends[i, :n]   = b["span_ends"]
        span_labels[i, :n] = b["span_labels"]
        span_mask[i, :n]   = True

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "span_starts":    span_starts,
        "span_ends":      span_ends,
        "span_labels":    span_labels,
        "span_mask":      span_mask,
    }


def build_span_datasets(
    data_dir: str | Path,
    tokenizer_name: str,
    max_length: int = 512,
    augment: bool = False,
    context_window: int = 0,
) -> tuple[SpanNERDataset, SpanNERDataset, SpanNERDataset]:
    data_dir  = Path(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_docs = load_split_docs(data_dir / "train.json")
    dev_docs   = load_split_docs(data_dir / "dev.json")
    test_docs  = load_split_docs(data_dir / "test.json")

    if context_window > 0:
        train_sents = build_context_sentences(train_docs, window=context_window)
        dev_sents   = build_context_sentences(dev_docs,   window=context_window)
        test_sents  = build_context_sentences(test_docs,  window=context_window)
        print(f"[context] window={context_window}: {len(train_sents)} train sentences")
    else:
        train_sents = [s for doc in train_docs for s in doc]
        dev_sents   = [s for doc in dev_docs   for s in doc]
        test_sents  = [s for doc in test_docs  for s in doc]

    if augment:
        flat     = [s for doc in train_docs for s in doc]
        aug_only = augment_entity_swap(flat)[len(flat):]
        train_sents = train_sents + aug_only
        print(f"[augment] +{len(aug_only)} sentences → {len(train_sents)} total")

    return (
        SpanNERDataset(tokenize_for_spans(train_sents, tokenizer, max_length)),
        SpanNERDataset(tokenize_for_spans(dev_sents,   tokenizer, max_length)),
        SpanNERDataset(tokenize_for_spans(test_sents,  tokenizer, max_length)),
    )


def compute_span_class_weights(dataset: SpanNERDataset) -> torch.Tensor:
    counts = torch.zeros(NUM_SPAN_CLASSES)
    for item in dataset:
        for lbl in item["span_labels"]:
            counts[lbl.item()] += 1
    total = counts.sum()
    return (total / (NUM_SPAN_CLASSES * counts.clamp(min=1))).sqrt()
