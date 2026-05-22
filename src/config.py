from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    model_name: str = "scibert_linear"
    base_model: str = "allenai/scibert_scivocab_cased"
    lr: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    max_length: int = 512
    seed: int = 42
    use_qlora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32


EXPERIMENTS = [
    ExperimentConfig(
        model_name="bert_linear",
        base_model="bert-base-cased",
    ),
    ExperimentConfig(
        model_name="scibert_linear",
        base_model="allenai/scibert_scivocab_cased",
    ),
    ExperimentConfig(
        model_name="scibert_mlp",
        base_model="allenai/scibert_scivocab_cased",
    ),
    ExperimentConfig(
        model_name="scibert_crf",
        base_model="allenai/scibert_scivocab_cased",
    ),
    ExperimentConfig(
        model_name="scibert_mlp_crf",
        base_model="allenai/scibert_scivocab_cased",
    ),
    ExperimentConfig(
        model_name="deberta_qlora",
        base_model="microsoft/deberta-v3-large",
        use_qlora=True,
        lora_rank=16,
        lora_alpha=32,
        batch_size=8,
    ),
]
