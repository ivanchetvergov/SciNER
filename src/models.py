import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoModel, BitsAndBytesConfig


class BertNER(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "bert-base-cased",
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(hidden))
        loss = None
        if labels is not None:
            w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100, weight=w)
        return loss, logits


class SciBertNER(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "allenai/scibert_scivocab_cased",
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(hidden))
        loss = None
        if labels is not None:
            w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100, weight=w)
        return loss, logits


class SciBertMLP(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "allenai/scibert_scivocab_cased",
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, num_labels),
        )
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.mlp(self.dropout(hidden))
        loss = None
        if labels is not None:
            w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100, weight=w)
        return loss, logits


class SciBertCRF(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "allenai/scibert_scivocab_cased",
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.register_buffer("class_weights", class_weights)

    def _apply_weights(self, emissions: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            emissions = emissions + torch.log(self.class_weights.to(emissions.device))
        return emissions

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self._apply_weights(self.classifier(self.dropout(hidden)).float())
        if labels is not None:
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            # Exclude context-window tokens (-100) from CRF; pos-0 must stay True
            mask = labels != -100
            mask[:, 0] = True
            loss = -self.crf(emissions, safe_labels, mask=mask, reduction="sum") / mask.sum()
            return loss, emissions
        return None, emissions

    def decode(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self._apply_weights(self.classifier(self.dropout(hidden)).float())
        return self.crf.decode(emissions, mask=attention_mask.bool())


class SciBertMLPCRF(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "allenai/scibert_scivocab_cased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, num_labels),
        )
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.mlp(self.dropout(hidden)).float()
        if labels is not None:
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            mask = attention_mask.bool()
            loss = -self.crf(emissions, safe_labels, mask=mask, reduction="sum") / mask.sum()
            return loss, emissions
        return None, emissions

    def decode(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.mlp(self.dropout(hidden)).float()
        return self.crf.decode(emissions, mask=attention_mask.bool())


class SciBertConcat4(nn.Module):
    """SciBERT with last-4-hidden-layers concatenation → Linear head."""

    def __init__(self, num_labels: int, base_model: str = "allenai/scibert_scivocab_cased",
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden * 4, num_labels)
        self.register_buffer("class_weights", class_weights)

        self._captured: list[torch.Tensor] = []
        for layer in self.encoder.encoder.layer[-4:]:
            layer.register_forward_hook(
                lambda m, inp, out: self._captured.append(out[0] if isinstance(out, (tuple, list)) else out)
            )

    def forward(self, input_ids, attention_mask, labels=None):
        self._captured.clear()
        self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_4 = torch.cat(self._captured, dim=-1)
        logits = self.classifier(self.dropout(last_4))
        loss = None
        if labels is not None:
            w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100, weight=w)
        return loss, logits


_CRF_MODELS = (SciBertCRF, SciBertMLPCRF)


def build_model(model_name: str, base_model: str, num_labels: int,
                use_qlora: bool = False, lora_rank: int = 16, lora_alpha: int = 32,
                class_weights: torch.Tensor | None = None) -> nn.Module:
    if model_name in ("bert_linear", "bert_linear_noweight", "roberta_linear", "roberta_linear_noweight"):
        return BertNER(num_labels, base_model, class_weights)
    if model_name in ("scibert_linear", "scibert_linear_noweight"):
        return SciBertNER(num_labels, base_model, class_weights)
    if model_name == "scibert_mlp":
        return SciBertMLP(num_labels, base_model, class_weights)
    if model_name in ("scibert_linear_crf", "scibert_linear_crf_noweight"):
        return SciBertCRF(num_labels, base_model, class_weights)
    if model_name == "scibert_concat4":
        return SciBertConcat4(num_labels, base_model, class_weights)
    if model_name == "deberta_qlora":
        return DeBertaQLoRA(num_labels, base_model, lora_rank, lora_alpha, class_weights)
    raise ValueError(f"Unknown model: {model_name}")


def is_crf_model(model: nn.Module) -> bool:
    inner = model.module if isinstance(model, nn.DataParallel) else model
    return isinstance(inner, _CRF_MODELS)
