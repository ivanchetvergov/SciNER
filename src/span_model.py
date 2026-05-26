"""Span-based NER: SciBERT encoder + (h_start; h_end; width_emb) → classifier.

Span representation follows SPERT (Eberts & Ulges, 2020):
  cat(h[start], h[end], width_embedding[span_len]) → Linear → num_span_classes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.span_data import MAX_SPAN_LEN, NUM_SPAN_CLASSES

WIDTH_EMB_DIM = 20


class SciBertSpan(nn.Module):
    def __init__(
        self,
        base_model: str = "allenai/scibert_scivocab_cased",
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(base_model)
        H               = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.width_emb  = nn.Embedding(MAX_SPAN_LEN + 1, WIDTH_EMB_DIM)
        self.classifier = nn.Linear(H * 2 + WIDTH_EMB_DIM, NUM_SPAN_CLASSES)
        self.register_buffer("class_weights", class_weights)

    def _span_repr(
        self,
        h: torch.Tensor,           # [B, T, H]
        span_starts: torch.Tensor, # [B, S]
        span_ends: torch.Tensor,   # [B, S]
    ) -> torch.Tensor:             # [B, S, 2H+W]
        H   = h.size(-1)
        h_s = h.gather(1, span_starts.unsqueeze(-1).expand(-1, -1, H))
        h_e = h.gather(1, span_ends.unsqueeze(-1).expand(-1, -1, H))
        w   = self.width_emb((span_ends - span_starts).clamp(0, MAX_SPAN_LEN))
        return torch.cat([h_s, h_e, w], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
        span_labels: torch.Tensor | None = None,
        span_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        h      = self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        logits = self.classifier(self._span_repr(h, span_starts, span_ends))  # [B, S, C]

        loss = None
        if span_labels is not None and span_mask is not None:
            w    = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits[span_mask], span_labels[span_mask], weight=w)

        return loss, logits
