import copy

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from .models import is_crf_model
from .utils import compute_metrics
from .data import ID2LABEL, ENTITY_TYPES


def _is_quantized(model) -> bool:
    return any("bitsandbytes" in type(m).__module__ for m in model.modules())


def train_epoch(model, loader: DataLoader, optimizer, device: torch.device,
                scheduler=None, grad_accum_steps: int = 1,
                scaler: GradScaler | None = None) -> float:
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with autocast(enabled=use_amp):
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if use_amp:
            scaler.scale(loss / grad_accum_steps).backward()
        else:
            (loss / grad_accum_steps).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    id2label: dict = None,
    return_preds: bool = False,
    use_amp: bool = False,
) -> dict | tuple:
    if id2label is None:
        id2label = ID2LABEL

    model.eval()
    all_preds, all_true = [], []
    use_crf = is_crf_model(model)

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with autocast(enabled=use_amp):
            if use_crf:
                decode = model.module.decode if isinstance(model, torch.nn.DataParallel) else model.decode
                tag_seqs = decode(input_ids, attention_mask)
                for tags, label_seq, mask_seq in zip(tag_seqs, labels, attention_mask):
                    pred_tags, true_tags = [], []
                    tag_idx = 0
                    for l, m in zip(label_seq, mask_seq):
                        if m.item() == 0:
                            break
                        if l.item() != -100:
                            pred_tags.append(id2label[tags[tag_idx]] if tag_idx < len(tags) else "O")
                            true_tags.append(id2label[l.item()])
                        tag_idx += 1
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

    metrics = compute_metrics(all_preds, all_true, ENTITY_TYPES)
    if return_preds:
        return metrics, all_preds, all_true
    return metrics


def train_model(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer,
    device: torch.device,
    num_epochs: int,
    model_name: str = "",
    patience: int = 0,
    scheduler=None,
    grad_accum_steps: int = 1,
    scaler: GradScaler | None = None,
) -> tuple[dict, list[dict]]:
    """
    Full training loop with best-checkpoint tracking by dev macro F1.
    patience > 0 enables early stopping.
    Quantized (4-bit) models skip deepcopy/load_state_dict — incompatible with bitsandbytes.
    Returns (best_metrics, history).
    """
    quantized = _is_quantized(model)
    use_amp = scaler is not None
    best_f1 = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, device,
                               scheduler, grad_accum_steps, scaler)
        dev_metrics = evaluate(model, dev_loader, device, use_amp=use_amp)
        dev_f1 = dev_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "dev_macro_f1": dev_f1,
        })

        improved = dev_f1 > best_f1
        tag = " *" if improved else ""
        print(f"[{model_name}] Epoch {epoch:2d} | loss: {avg_loss:.4f} | dev macro F1: {dev_f1:.4f}{tag}")

        if improved:
            best_f1 = dev_f1
            no_improve = 0
            if not quantized:
                inner = model.module if isinstance(model, torch.nn.DataParallel) else model
                best_state = copy.deepcopy(inner.state_dict())
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"[{model_name}] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        inner = model.module if isinstance(model, torch.nn.DataParallel) else model
        inner.load_state_dict(best_state)

    return {"best_dev_macro_f1": best_f1}, history
