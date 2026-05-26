import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


ENTITY_TYPES = ["Generic", "Material", "Method", "Metric", "OtherScientificTerm", "Task"]

LABEL_LIST = ["O"] + [f"B-{t}" for t in ENTITY_TYPES] + [f"I-{t}" for t in ENTITY_TYPES]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)


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


def load_split(path: str | Path) -> list[list[tuple[str, str]]]:
    docs = load_scierc(path)
    sentences = []
    for doc in docs:
        sentences.extend(doc_to_bio_sentences(doc))
    return sentences


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
            padding="max_length",
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(LABEL2ID[word_labels[word_id]])
            else:
                aligned_labels.append(-100)
            prev_word_id = word_id

        dataset.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": aligned_labels,
        })

    return dataset


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


def build_datasets(
    data_dir: str | Path,
    tokenizer_name: str,
    max_length: int = 512,
) -> tuple[NERDataset, NERDataset, NERDataset]:
    data_dir = Path(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_sents = load_split(data_dir / "train.json")
    dev_sents   = load_split(data_dir / "dev.json")
    test_sents  = load_split(data_dir / "test.json")

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
    return total / (num_labels * counts.clamp(min=1))
