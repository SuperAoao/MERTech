import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from config_cbf import URL, NUM_LABELS


class SSLNetCBF_IPT(nn.Module):
    """Single-task IPT model (7 classes) + optional onset auxiliary output."""

    def __init__(self, url=URL, weight_sum=True, freeze_all=False):
        super().__init__()
        self.url = url
        encode_size = 24 if "330M" in self.url else 12
        self.frontend = HuggingfaceFrontendCBF(
            url=self.url, use_last=(1 - int(weight_sum)), encoder_size=encode_size, freeze_all=freeze_all
        )
        # one-hidden-layer MLP head (512 units) implemented by BackendCBF
        self.backend_tech = BackendCBF(NUM_LABELS, encoder_size=encode_size)

        # auxiliary onset head (kept because existing MERTech uses it; harmless if unused in loss)
        self.backend_onset = BackendCBF(1, encoder_size=encode_size)

    def forward(self, x):
        # x: [B, 1, L] or [B, L]
        if x.dim() == 3:
            x = x.squeeze(dim=1)

        h = self.frontend(x)  # [B, T_mert, D]
        tech_logits_btC, _ = self.backend_tech(h)  # [B, T_mert, 7]
        onset_logits_bt1, _ = self.backend_onset(h)  # [B, T_mert, 1]

        # return [B, C, T] for consistency with existing evaluation code
        return tech_logits_btC.transpose(-1, -2), onset_logits_bt1.transpose(-1, -2)


class HuggingfaceFrontendCBF(nn.Module):
    def __init__(self, url, use_last=False, encoder_size=12, freeze_all=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(url, trust_remote_code=True)
        if freeze_all:
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            # keep HF feature extractor frozen (common for wav2vec-style models)
            if hasattr(self.model, "feature_extractor"):
                self.model.feature_extractor._freeze_parameters()

        self.use_last = use_last
        if encoder_size == 12:
            self.layer_weights = nn.Parameter(torch.ones(13))
        elif encoder_size == 24:
            self.layer_weights = nn.Parameter(torch.ones(25))
        else:
            raise ValueError("encoder_size must be 12 or 24")

    def forward(self, x):
        out = self.model(x, output_hidden_states=True)
        if self.use_last:
            h = out["last_hidden_state"]  # [B, T, D]
            h = F.pad(h, (0, 0, 0, 1), mode="reflect")
            return h

        hs = out["hidden_states"]  # tuple/list of [B, T, D]
        h = torch.stack(hs, dim=3)  # [B, T, D, L]
        h = F.pad(h, (0, 0, 0, 0, 0, 1), mode="reflect")
        w = torch.softmax(self.layer_weights, dim=0)  # [L]
        # weighted sum over layer dimension
        h = torch.matmul(h, w)  # [B, T, D]
        return h


class BackendCBF(nn.Module):
    def __init__(self, class_size, encoder_size=12, frame=True):
        super().__init__()
        if encoder_size == 12:
            feature_dim = 768
        elif encoder_size == 24:
            feature_dim = 1024
        else:
            raise ValueError("encoder_size must be 12 or 24")
        hidden_dim = 512
        self.frame = frame
        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, class_size)

    def forward(self, x):
        x = self.proj(x)
        if not self.frame:
            x = x.mean(1, keepdim=False)
        feature = x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x, feature


class SelfAttnCBF(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        y, _ = self.att(x, x, x)
        return y + x


class BackendAttnCBF(nn.Module):
    def __init__(self, class_size, feature_dim):
        super().__init__()
        self.attn = SelfAttnCBF(feature_dim, num_heads=1)
        self.proj = nn.Linear(feature_dim, class_size)

    def forward(self, x):
        x = self.attn(x)
        return self.proj(x)


# Backwards-compatible alias (older name used in first draft)
SSLNetCBF = SSLNetCBF_IPT

