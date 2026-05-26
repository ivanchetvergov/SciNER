import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer


ENTITY_TYPES = ["Generic", "Material", "Method", "Metric", "OtherScientificTerm", "Task"]

LABEL_LIST = ["O"] + [f"B-{t}" for t in ENTITY_TYPES] + [f"I-{t}" for t in ENTITY_TYPES]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)

_CTX = "__ctx__"  # sentinel label for context tokens → -100


def load_scierc(path: str | Path) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def doc_to_bio_sentences(doc: dict) -> list[list[tuple[str, str]]]:
    sentences = doc["sentences"]
    ner_per_sentence = doc["ner"]

    offsets = []
    current = 0
    for sent in sentences:
        offsets.append(current)
        current += len(sent)

    result = []
    for sent_idx, (sent, spans) in enumerate(zip(sentences, ner_per_sentence)):
        offset = offsets[sent_idx]
        n = len(sent)
        labels = ["O"] * n

        for start, end, ent_type in spans:
            local_start = start - offset
            local_end = end - offset
            if local_start < 0 or local_end >= n:
                continue
            labels[local_start] = f"B-{ent_type}"
            for i in range(local_start + 1, local_end + 1):
                labels[i] = f"I-{ent_type}"

        result.append(list(zip(sent, labels)))

    return result


def load_split_docs(path: str | Path) -> list[list[list[tuple[str, str]]]]:
    """Load preserving document structure: list[doc] → list[sent] → list[(word, label)]."""
    return [doc_to_bio_sentences(doc) for doc in load_scierc(path)]


def load_split(path: str | Path) -> list[list[tuple[str, str]]]:
    return [sent for doc in load_split_docs(path) for sent in doc]


# ── cross-sentence context ─────────────────────────────────────────────────

def build_context_sentences(
    docs: list[list[list[tuple[str, str]]]],
    window: int = 1,
) -> list[list[tuple[str, str]]]:
    """
    For each sentence, prepend/append up to `window` neighboring sentences
    from the same document. Neighboring tokens are labeled _CTX → -100,
    so loss is computed only on the center sentence.
    """
    result = []
    for doc in docs:
        for i, sent in enumerate(doc):
            before = [
                (w, _CTX)
                for j in range(max(0, i - window), i)
                for w, _ in doc[j]
            ]
            after = [
                (w, _CTX)
                for j in range(i + 1, min(len(doc), i + window + 1))
                for w, _ in doc[j]
            ]
            result.append(before + sent + after)
    return result


# ── entity swap augmentation ───────────────────────────────────────────────

def _extract_entity_spans(
    sent: list[tuple[str, str]],
) -> list[tuple[int, int, str, tuple[str, ...]]]:
    spans = []
    i = 0
    while i < len(sent):
        word, label = sent[i]
        if label.startswith("B-"):
            etype = label[2:]
            j = i + 1
            while j < len(sent) and sent[j][1] == f"I-{etype}":
                j += 1
            spans.append((i, j - 1, etype, tuple(w for w, _ in sent[i:j])))
            i = j
        else:
            i += 1
    return spans


def augment_entity_swap(
    sentences: list[list[tuple[str, str]]],
    p: float = 0.5,
    seed: int = 42,
) -> list[list[tuple[str, str]]]:
    """
    Returns original sentences + augmented copies (up to 2x dataset size).
    Each entity span is replaced with probability p by a random entity of
    the same type from the training pool.
    """
    rng = random.Random(seed)

    pool: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for sent in sentences:
        for _, _, etype, words in _extract_entity_spans(sent):
            pool[etype].append(words)

    augmented = []
    for sent in sentences:
        spans = _extract_entity_spans(sent)
        if not spans:
            continue

        new_sent = list(sent)
        offset = 0
        swapped = False

        for start, end, etype, words in spans:
            candidates = [e for e in pool[etype] if e != words]
            if not candidates or rng.random() > p:
                continue
            new_words = rng.choice(candidates)
            new_labels = [f"B-{etype}"] + [f"I-{etype}"] * (len(new_words) - 1)
            s, e_ = start + offset, end + offset
            new_sent = new_sent[:s] + list(zip(new_words, new_labels)) + new_sent[e_ + 1:]
            offset += len(new_words) - (end - start + 1)
            swapped = True

        if swapped:
            augmented.append(new_sent)

    return sentences + augmented


# ── tokenisation ───────────────────────────────────────────────────────────

def tokenize_and_align_labels(
    sentences: list[list[tuple[str, str]]],
    tokenizer,
    max_length: int = 512,
) -> list[dict]:
    dataset = []
    for sent in sentences:
        tokens = [t for t, _ in sent]
        word_labels = [l for _, l in sent]

        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                label = word_labels[word_id]
                aligned_labels.append(-100 if label == _CTX else LABEL2ID[label])
            else:
                label = word_labels[word_id]
                if label == _CTX:
                    aligned_labels.append(-100)
                elif label.startswith("B-"):
                    aligned_labels.append(LABEL2ID["I-" + label[2:]])
                else:
                    aligned_labels.append(LABEL2ID[label])
            prev_word_id = word_id

        dataset.append({
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels":         aligned_labels,
        })

    return dataset


# ── dataset & collate ──────────────────────────────────────────────────────

class NERDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            "input_ids":      torch.tensor(item["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(item["labels"],         dtype=torch.long),
        }


def ner_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids":      pad_sequence([b["input_ids"]      for b in batch], batch_first=True, padding_value=0),
        "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
        "labels":         pad_sequence([b["labels"]         for b in batch], batch_first=True, padding_value=-100),
    }


def build_datasets(
    data_dir: str | Path,
    tokenizer_name: str,
    max_length: int = 512,
    augment: bool = False,
    context_window: int = 0,
) -> tuple[NERDataset, NERDataset, NERDataset]:
    data_dir = Path(data_dir)
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
        flat = [s for doc in train_docs for s in doc]
        aug_only = augment_entity_swap(flat)[len(flat):]
        train_sents = train_sents + aug_only
        print(f"[augment] entity swap: +{len(aug_only)} sentences → {len(train_sents)} total")

    train_data = tokenize_and_align_labels(train_sents, tokenizer, max_length)
    dev_data   = tokenize_and_align_labels(dev_sents,   tokenizer, max_length)
    test_data  = tokenize_and_align_labels(test_sents,  tokenizer, max_length)

    return NERDataset(train_data), NERDataset(dev_data), NERDataset(test_data)


def compute_class_weights(dataset: NERDataset, num_labels: int) -> torch.Tensor:
    counts = torch.zeros(num_labels)
    for item in dataset:
        for l in item["labels"]:
            if l.item() != -100:
                counts[l.item()] += 1
    total = counts.sum()
    weights = total / (num_labels * counts.clamp(min=1))
    return weights.sqrt()
