import copy

import torch
from torch.utils.data import DataLoader

from .models import is_crf_model
from .utils import compute_metrics
from .data import ID2LABEL, ENTITY_TYPES


def train_epoch(model, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    id2label: dict = None,
) -> dict:
    if id2label is None:
        id2label = ID2LABEL

    model.eval()
    all_preds, all_true = [], []
    use_crf = is_crf_model(model)

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        if use_crf:
            tag_seqs = model.decode(input_ids, attention_mask)
            # align with label positions (skip -100)
            for tags, label_seq in zip(tag_seqs, labels):
                pred_tags, true_tags = [], []
                tag_iter = iter(tags)
                for l in label_seq:
                    if l.item() == -100:
                        continue
                    true_tags.append(id2label[l.item()])
                    try:
                        pred_tags.append(id2label[next(tag_iter)])
                    except StopIteration:
                        pred_tags.append("O")
                all_preds.append(pred_tags)
                all_true.append(true_tags)
        else:
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = logits.argmax(dim=-1)
            for pred_seq, label_seq in zip(pred_ids, labels):
                pred_tags, true_tags = [], []
                for p, l in zip(pred_seq, label_seq):
                    if l.item() == -100:
                        continue
                    pred_tags.append(id2label[p.item()])
                    true_tags.append(id2label[l.item()])
                all_preds.append(pred_tags)
                all_true.append(true_tags)

    return compute_metrics(all_preds, all_true, ENTITY_TYPES)


def train_model(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer,
    device: torch.device,
    num_epochs: int,
    model_name: str = "",
) -> tuple[dict, list[dict]]:
    """
    Full training loop with best-checkpoint tracking by dev macro F1.
    Returns (best_metrics, history).
    history is a list of {epoch, train_loss, dev_macro_f1} dicts.
    """
    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        dev_metrics = evaluate(model, dev_loader, device)
        dev_f1 = dev_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "dev_macro_f1": dev_f1,
        })

        tag = " *" if dev_f1 > best_f1 else ""
        print(f"[{model_name}] Epoch {epoch:2d} | loss: {avg_loss:.4f} | dev macro F1: {dev_f1:.4f}{tag}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_dev_macro_f1": best_f1}, history
