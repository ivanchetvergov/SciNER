from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    model_name: str = "scibert_linear"
    base_model: str = "allenai/scibert_scivocab_cased"
    lr: float = 2e-5
    crf_lr: float = 0.0          # if > 0, CRF layer uses this lr; encoder uses lr
    batch_size: int = 16
    num_epochs: int = 10
    early_stopping_patience: int = 0   # 0 = disabled
    max_length: int = 512
    seed: int = 42
    use_qlora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32


EXPERIMENTS = [
    ExperimentConfig(
        model_name="bert_linear",
        base_model="bert-base-cased",
        num_epochs=5,
    ),
    ExperimentConfig(
        model_name="scibert_linear",
        base_model="allenai/scibert_scivocab_cased",
        num_epochs=5,
    ),
    ExperimentConfig(
        model_name="scibert_mlp",
        base_model="allenai/scibert_scivocab_cased",
        num_epochs=5,
    ),
    ExperimentConfig(
        model_name="scibert_crf",
        base_model="allenai/scibert_scivocab_cased",
        num_epochs=8,
        early_stopping_patience=3,
        crf_lr=1e-3,
    ),
    ExperimentConfig(
        model_name="scibert_mlp_crf",
        base_model="allenai/scibert_scivocab_cased",
        num_epochs=8,
        early_stopping_patience=3,
        crf_lr=1e-3,
    ),
    ExperimentConfig(
        model_name="scibert_concat4",
        base_model="allenai/scibert_scivocab_cased",
        num_epochs=5,
    ),
    ExperimentConfig(
        model_name="deberta_qlora",
        base_model="microsoft/deberta-v3-large",
        use_qlora=True,
        lora_rank=16,
        lora_alpha=32,
        batch_size=8,
        num_epochs=5,
    ),
]
