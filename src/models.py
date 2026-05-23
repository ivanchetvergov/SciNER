import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, BitsAndBytesConfig


class BertNER(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "bert-base-cased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits


class SciBertNER(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "allenai/scibert_scivocab_cased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits


class SciBertMLP(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "allenai/scibert_scivocab_cased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.mlp(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
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
            loss = -self.crf(emissions, safe_labels, mask=attention_mask.bool(), reduction="mean")
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
            loss = -self.crf(emissions, safe_labels, mask=attention_mask.bool(), reduction="mean")
            return loss, emissions
        return None, emissions

    def decode(self, input_ids, attention_mask):
        emissions = self.mlp(
            self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        )
        return self.crf.decode(emissions, mask=attention_mask.bool())


class DeBertaQLoRA(nn.Module):
    def __init__(self, num_labels: int, base_model: str = "microsoft/deberta-v3-large",
                 lora_rank: int = 16, lora_alpha: int = 32):
        super().__init__()
        from peft import get_peft_model, LoraConfig, TaskType

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModel.from_pretrained(base_model, quantization_config=bnb_config)
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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        logits = self.classifier.to(hidden.device)(hidden)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits


_CRF_MODELS = (SciBertCRF, SciBertMLPCRF)


def build_model(model_name: str, base_model: str, num_labels: int,
                use_qlora: bool = False, lora_rank: int = 16, lora_alpha: int = 32) -> nn.Module:
    if model_name == "bert_linear":
        return BertNER(num_labels, base_model)
    if model_name == "scibert_linear":
        return SciBertNER(num_labels, base_model)
    if model_name == "scibert_mlp":
        return SciBertMLP(num_labels, base_model)
    if model_name == "scibert_crf":
        return SciBertCRF(num_labels, base_model)
    if model_name == "scibert_mlp_crf":
        return SciBertMLPCRF(num_labels, base_model)
    if model_name == "deberta_qlora":
        return DeBertaQLoRA(num_labels, base_model, lora_rank, lora_alpha)
    raise ValueError(f"Unknown model: {model_name}")


def is_crf_model(model: nn.Module) -> bool:
    return isinstance(model, _CRF_MODELS)
