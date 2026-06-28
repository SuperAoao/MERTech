import torch.nn.functional as F
import sys
sys.path.append('../fun')
import torch
from typing import Optional
from transformers import AutoModel
from torch import nn
import config

class TemporalPyramidTransformer(nn.Module):
    """
    Option A: produce x_fused with same shape as input x: [B, T, D].

    - Build a temporal pyramid by strided conv downsampling over time.
    - Apply Transformer encoder at each level (cheaper at coarse levels).
    - Fuse coarse->fine via upsample + residual add.
    """

    def __init__(
        self,
        d_model: int,
        num_levels: int = 3,
        num_layers_per_level: int = 1,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.num_levels = num_levels

        # Downsample along time (T) while keeping channel dim (D).
        # Operates on [B, D, T] for Conv1d.
        self.down = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
                for _ in range(num_levels - 1)
            ]
        )

        # Per-level Transformer encoders over [B, T_level, D].
        self.encoders = nn.ModuleList()
        for _ in range(num_levels):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoders.append(nn.TransformerEncoder(layer, num_layers=num_layers_per_level))

        # Light normalization + residual gating for stability.
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.res_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] -> x_fused: [B, T, D]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to be [B, T, D], got shape {tuple(x.shape)}")

        x0 = self.norm_in(x)
        feats = [x0]

        # Build pyramid
        h = x0.transpose(1, 2)  # [B, D, T]
        for down in self.down:
            h = down(h)  # [B, D, T/2]
            feats.append(h.transpose(1, 2))  # [B, T_level, D]

        # Intra-level encoding
        feats = [enc(f) for enc, f in zip(self.encoders, feats)]

        # Top-down fusion (coarse -> fine)
        fused = feats[-1]
        for lvl in range(self.num_levels - 2, -1, -1):
            target_len = feats[lvl].shape[1]
            # Upsample coarse sequence to match fine length.
            up = F.interpolate(
                fused.transpose(1, 2), size=target_len, mode="linear", align_corners=False
            ).transpose(1, 2)
            fused = feats[lvl] + up

        # Residual: keep original x accessible; gate controls how much FPT changes x.
        fused = self.norm_out(fused) # When FPT is used, the fused feature is normalized before the residual connection.
        x_fused = x + torch.tanh(self.res_gate) * fused # When res_gate is 0, the fused feature is not used.
        return x_fused


class PNHead(nn.Module):
    """
    Dedicated PN (dianyin) frame classifier with local temporal context and
    optional pluck/onset conditioning. Output logits [B, T].
    """

    def __init__(
        self,
        d_model: int,
        context: int = 5,
        hidden: int = 256,
        dropout: float = 0.2,
        use_pluck_gate: bool = True,
        use_onset_gate: bool = False,
    ):
        super().__init__()
        self.use_pluck_gate = use_pluck_gate
        self.use_onset_gate = use_onset_gate
        kernel = 2 * context + 1
        self.ctx_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=context)
        if use_pluck_gate:
            self.pluck_gate = nn.Linear(1, d_model)
        if use_onset_gate:
            self.onset_gate = nn.Linear(1, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pluck_prob: Optional[torch.Tensor] = None,
        onset_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [B, T, D]
        h = self.ctx_conv(x.transpose(1, 2)).transpose(1, 2)
        if self.use_pluck_gate and pluck_prob is not None:
            if pluck_prob.dim() == 3:
                pluck_prob = pluck_prob.squeeze(-1)
            h = h + self.pluck_gate(pluck_prob.unsqueeze(-1))
        if self.use_onset_gate and onset_prob is not None:
            if onset_prob.dim() == 3:
                onset_prob = onset_prob.squeeze(-1)
            h = h + self.onset_gate(onset_prob.unsqueeze(-1))
        return self.classifier(h).squeeze(-1)


class SSLNet(nn.Module):
    def __init__(self,
                 url,
                 class_num,
                 weight_sum=False,
                 freeze_all=False
                 ):
        super().__init__()
        self.num_classes = class_num
        self.url = url
        encode_size = 24 if "330M" in self.url else 12
        self.frontend = HuggingfaceFrontend(url=self.url, use_last=(1-weight_sum), encoder_size=encode_size, freeze_all=freeze_all)

        # Optional Feature Pyramid Transformer (Option A): x -> x_fused (same shape)
        feature_dim = 1024 if encode_size == 24 else 768
        self.fpt = None
        if config.USE_FPT:
            self.fpt = TemporalPyramidTransformer(
                d_model=feature_dim,
                num_levels=config.FPT_LEVELS,
                num_layers_per_level=config.FPT_NUM_LAYERS,
                nhead=config.FPT_NUM_HEADS,
                dropout=config.FPT_DROPOUT,
            )

        self.backend = Backend(class_num, encoder_size=encode_size)
        self.backend_onset = Backend(1, encoder_size=encode_size)
        self.backend_attnet_IPT =Backend_Attnet(config.NUM_LABELS,config.NUM_LABELS+1)
        self.backend_attnet_pitch = Backend_Attnet(config.MAX_MIDI-config.MIN_MIDI+1,config.MAX_MIDI-config.MIN_MIDI+1+1)

        self.pn_head = None
        if config.USE_PN_HEAD:
            self.pn_head = PNHead(
                d_model=feature_dim,
                context=config.PN_HEAD_CONTEXT,
                hidden=config.PN_HEAD_HIDDEN,
                dropout=config.PN_HEAD_DROPOUT,
                use_pluck_gate=config.PN_HEAD_USE_PLUCK_GATE,
                use_onset_gate=config.PN_HEAD_USE_ONSET_GATE,
            )
        self._pn_aux_logit = None

    def forward(self, x):
        x = x.squeeze(dim=1)
        x_mert = self.frontend(x)  # [B, T_mert, D]
        x_task = x_mert
        if self.fpt is not None:
            x_task = self.fpt(x_mert)  # [B, T_mert, D] IPT / pitch / PN head features

        out, feature = self.backend(x_task) #[batch, time, class_num]
        sizes = out.size()
        out = out.view(sizes[0], sizes[1], config.NUM_LABELS, config.MAX_MIDI - config.MIN_MIDI + 1)
        IPT_pred =  torch.sum(out, dim=3)  # [batch, time, IPT]
        pitch_pred = torch.sum(out, dim=2) # [batch, time, pitch]

        use_onset_bypass = (
            self.fpt is not None and config.USE_FPT_ONSET_BYPASS
        )
        x_onset = x_mert if use_onset_bypass else x_task
        onset_pred, _ = self.backend_onset(x_onset) #[batch, time, class_num]
        onset_pred_deta = onset_pred.detach()

        IPT_onset_cat = torch.cat((IPT_pred, onset_pred_deta), 2)
        pitch_onset_cat = torch.cat((pitch_pred, onset_pred_deta), 2)

        IPT_pred_out = self.backend_attnet_IPT(IPT_onset_cat).transpose(-1,-2)
        pitch_pred_out = self.backend_attnet_pitch(pitch_onset_cat).transpose(-1,-2)
        onset_pred_out = onset_pred.transpose(-1,-2)

        self._pn_aux_logit = None
        if self.pn_head is not None:
            pluck_prob = torch.sigmoid(IPT_pred[:, :, config.PLUCKS_IDX].detach())
            onset_prob = None
            if config.PN_HEAD_USE_ONSET_GATE:
                onset_prob = torch.sigmoid(onset_pred_deta.squeeze(-1))
            pn_aux = self.pn_head(x_task, pluck_prob, onset_prob)
            self._pn_aux_logit = pn_aux
            fused_pn = (
                config.PN_FUSION_ALPHA * pn_aux
                + (1.0 - config.PN_FUSION_ALPHA) * IPT_pred_out[:, config.PN_IDX, :]
            )
            IPT_pred_out = IPT_pred_out.clone()
            IPT_pred_out[:, config.PN_IDX, :] = fused_pn

        return IPT_pred_out,pitch_pred_out,onset_pred_out


class HuggingfaceFrontend(nn.Module):
    def __init__(self, url, use_last=False, encoder_size=12, freeze_all=False):
        super().__init__()
        print("url is：",url)
        self.model = AutoModel.from_pretrained(config.URL, trust_remote_code=True)
        if freeze_all:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.feature_extractor._freeze_parameters()

        self.use_last = use_last
        if encoder_size == 12:
            self.layer_weights = torch.nn.parameter.Parameter(data=torch.ones(13), requires_grad=True)
        elif encoder_size == 24:
            self.layer_weights = torch.nn.parameter.Parameter(data=torch.ones(25), requires_grad=True)

    def forward(self,x):
        x = self.model(x, output_hidden_states=True)
        if self.use_last:
            h = x["last_hidden_state"]
            pad_width = (0, 0, 0, 1)
            h = F.pad(h, pad_width, mode='reflect')
        else:
            h = x["hidden_states"]
            h = torch.stack(h, dim=3)
            pad_width = (0, 0, 0, 0, 0, 1)
            h = F.pad(h, pad_width, mode='reflect')
        if not self.use_last:
            weights = torch.softmax(self.layer_weights,dim=0)
            h = torch.matmul(h, weights)
        return h

class Backend(nn.Module):
    def __init__(self, class_size, encoder_size=12, frame=True) -> None:
        super().__init__()
        assert encoder_size == 12 or encoder_size == 24
        if encoder_size == 12:
            self.feature_dim = 768
        elif encoder_size == 24:
            self.feature_dim = 1024
        else:
            raise NotImplementedError
        self.hidden_dim = 512
        self.proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.hidden_dim, class_size)
        self.frame = frame

    def forward(self, x):
        x = self.proj(x)
        if not self.frame:
            x = x.mean(1, False)
        feature = x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x, feature

class self_attn(nn.Module):
    def __init__(self, embeded_dim, num_heads):
        super(self_attn, self).__init__()
        self.att = nn.MultiheadAttention(embeded_dim, num_heads, batch_first=True)

    def forward(self, x):
        x1 = x #[batch, T/9, FRE*3]
        res_branch, attn_wei = self.att(x1, x1, x1)
        res = torch.add(res_branch, x)
        return res

class Backend_Attnet(nn.Module):
    def __init__(self, class_size, feature_dim):
        super().__init__()
        self.Attn = self_attn(feature_dim, 1)
        self.proj = nn.Linear(feature_dim, class_size)

    def forward(self, x):
        x = self.Attn(x)
        x = self.proj(x)
        return x