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
        self.classifier = nn.Linear(hidden, num_labels)
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)
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
        self.classifier = nn.Linear(hidden, num_labels)
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)
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
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, num_labels),
        )
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.mlp(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100, weight=w)
        return loss, logits


class SciBertCRF(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "allenai/scibert_scivocab_cased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        emissions = self.classifier(
            self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        )
        if labels is not None:
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            mask = attention_mask.bool()
            loss = -self.crf(emissions, safe_labels, mask=mask, reduction="sum") / mask.sum()
            return loss, emissions
        return None, emissions

    def decode(self, input_ids, attention_mask):
        emissions = self.classifier(
            self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        )
        return self.crf.decode(emissions, mask=attention_mask.bool())


class SciBertMLPCRF(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "allenai/scibert_scivocab_cased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, num_labels),
        )
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        emissions = self.mlp(
            self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        )
        if labels is not None:
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            mask = attention_mask.bool()
            loss = -self.crf(emissions, safe_labels, mask=mask, reduction="sum") / mask.sum()
            return loss, emissions
        return None, emissions

    def decode(self, input_ids, attention_mask):
        emissions = self.mlp(
            self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        )
        return self.crf.decode(emissions, mask=attention_mask.bool())


class SciBertConcat4(nn.Module):
    """SciBERT with last-4-hidden-layers concatenation → Linear head."""

    def __init__(self, num_labels: int, base_model: str = "allenai/scibert_scivocab_cased",
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model, output_hidden_states=True)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden * 4, num_labels)
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_4 = torch.cat(outputs.hidden_states[-4:], dim=-1)
        logits = self.classifier(last_4)
        loss = None
        if labels is not None:
            w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100, weight=w)
        return loss, logits


class DeBertaQLoRA(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "microsoft/deberta-v3-large",
                 lora_rank: int = 16, lora_alpha: int = 32,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModel.from_pretrained(base_model, quantization_config=bnb_config)
        base = prepare_model_for_kbit_training(base)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        self.encoder = get_peft_model(base, lora_config)
        self.classifier = nn.Linear(base.config.hidden_size, num_labels)
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        logits = self.classifier.to(hidden.device)(hidden)
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
    if model_name == "bert_linear":
        return BertNER(num_labels, base_model, class_weights)
    if model_name == "scibert_linear":
        return SciBertNER(num_labels, base_model, class_weights)
    if model_name == "scibert_mlp":
        return SciBertMLP(num_labels, base_model, class_weights)
    if model_name == "scibert_crf":
        return SciBertCRF(num_labels, base_model)
    if model_name == "scibert_mlp_crf":
        return SciBertMLPCRF(num_labels, base_model)
    if model_name == "scibert_concat4":
        return SciBertConcat4(num_labels, base_model, class_weights)
    if model_name == "deberta_qlora":
        return DeBertaQLoRA(num_labels, base_model, lora_rank, lora_alpha, class_weights)
    raise ValueError(f"Unknown model: {model_name}")


def is_crf_model(model: nn.Module) -> bool:
    inner = model.module if isinstance(model, nn.DataParallel) else model
    return isinstance(inner, _CRF_MODELS)
