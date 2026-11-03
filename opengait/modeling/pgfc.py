"""
PGFC - Part-wise Gated Feature Correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNormCanon(nn.Module):

    def __init__(self, channels: int, parts_num: int = 16, parallel_BN1d: bool = True):
        super().__init__()
        self.C = channels
        self.P = parts_num
        self.parallel_BN1d = parallel_BN1d
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(channels * parts_num)
        else:
            self.bn1d = nn.ModuleList(
                [nn.BatchNorm1d(channels) for _ in range(parts_num)]
            )

    def forward(self, x: torch.Tensor):
        """x: [B, C, P] -> y: [B, C, P] (batch-normalized)."""
        B, C, P = x.shape
        assert C == self.C and P == self.P, (
            f"Expected [B,{self.C},{self.P}], got [{B},{C},{P}]"
        )
        if self.parallel_BN1d:
            y = self.bn1d(x.reshape(B, C * P)).reshape(B, C, P)
        else:
            y = torch.cat(
                [bn(_x.squeeze(2)).unsqueeze(2)
                 for _x, bn in zip(x.split(1, 2), self.bn1d)],
                dim=2,
            )
        return y.contiguous()


class PartGateOnly(nn.Module):

    def __init__(self, channels: int, parts_num: int = 16, alpha: float = 0.5):
        super().__init__()
        self.C = channels
        self.P = parts_num
        self.alpha = alpha

        # Sample-adaptive part-wise gate head. Linear(P -> P) on the channel-pooled
        # part descriptor. Zero-init => identity start (s == 1).
        self.gate_fc = nn.Linear(parts_num, parts_num)
        nn.init.zeros_(self.gate_fc.weight)
        nn.init.zeros_(self.gate_fc.bias)

        # Diagnostics (populated each forward, no grad)
        self._last_scale_mean: float = 1.0
        self._last_scale_std: float = 0.0

    def forward(self, h: torch.Tensor, iter_num=None):
        """h: [B, C, P] -> (y: [B, C, P], info dict).

        iter_num is accepted only for call-site symmetry with GatedPGFC; this
        module has no schedule and ignores it.
        """
        B, C, P = h.shape
        assert C == self.C, f"Expected C={self.C}, got {C}"
        assert P == self.P, f"Expected P={self.P}, got {P}"

        # 1. Part descriptor: pool over the channel dimension.
        z = h.mean(dim=1)                       # [B, P]

        # 2. Sample-adaptive part-wise gate.
        m = torch.sigmoid(self.gate_fc(z))      # [B, P]

        # 3. Residual scale centered at 1: s in [1-alpha, 1+alpha].
        s = 1.0 + self.alpha * (2.0 * m - 1.0)  # [B, P]

        # 4. Part-wise reweighting of the ORIGINAL feature (broadcast over C).
        y = h * s.unsqueeze(1)                   # [B, C, P]
        y = y.contiguous()

        # 5. Diagnostics (detached, cheap). Reuse the PGFC logging keys so the
        # existing DeepGaitV2 part-vector logger can visualize the per-part scale.
        with torch.no_grad():
            scale_mean = s.mean().item()
            scale_std = s.std().item()
            scale_part_mean = s.mean(dim=0).detach().clone()   # [P]
            scale_part_mean_str = "[" + ", ".join(
                f"{v:.4f}" for v in scale_part_mean.float().cpu().tolist()
            ) + "]"
            self._last_scale_mean = scale_mean
            self._last_scale_std = scale_std

        info = {
            "pgfc_gate_mean": scale_mean,
            "pgfc_gate_std": scale_std,
            "pgfc_gate_part_mean": scale_part_mean,
            "pgfc_gate_part_mean_str": scale_part_mean_str,
            "gate_part_mean": scale_part_mean,
        }
        return y, info


class GatedPGFC(nn.Module):

    def __init__(
        self,
        channels: int,
        groups: int = 16,
        eps: float = 1e-4,
        var_floor: float = 1e-4,
        parts_num: int = 16,
        gate_init_bias: float = -4.0,
        gate_type: str = "bnneck",
        gate_granularity: str = "sample_part",
        use_gate: bool = True,
        gate_const: float = 1.0,
        stopgrad_gate_ctx: bool = True,
        affine_type: str = "group_part",
        stats_axes: str = "cg_only_partwise",
        # Gate cap schedule
        g_max_init: float = 0.1,
        g_max_final: float = 1.0,
        g_max_ramp_iters: int = 5000,
        # MixStyle (optional DG add-on)
        use_mixstyle: bool = False,
        mix_p: float = 0.5,
        mix_alpha: float = 0.1,
        # Gate regularisation (optional, default OFF)
        use_gate_reg: bool = False,
        gate_reg_target: float = 0.2,
        gate_reg_weight: float = 1e-4,
        # Analysis-only export (default OFF; never touched by train/eval)
        return_analysis: bool = False,
    ):
        super().__init__()
        assert channels % groups == 0, f"channels({channels}) must be divisible by groups({groups})"

        self.C = channels
        self.G = groups
        self.Cg = channels // groups
        self.P = parts_num
        self.eps = eps
        self.var_floor = var_floor
        self.gate_type = gate_type
        assert gate_granularity in ("sample_part", "sample", "part", "group_part"), (
            f"Unknown gate_granularity: {gate_granularity}. "
            "Expected one of 'sample_part', 'sample', 'part', 'group_part'."
        )
        self.gate_granularity = gate_granularity
        self.use_gate = use_gate
        self.gate_const = gate_const
        self.stopgrad_gate_ctx = stopgrad_gate_ctx
        self.affine_type = affine_type
        self.stats_axes = stats_axes

        # Gate cap schedule
        self.g_max_init = g_max_init
        self.g_max_final = g_max_final
        self.g_max_ramp_iters = g_max_ramp_iters

        # MixStyle config
        self.use_mixstyle = use_mixstyle
        self.mix_p = mix_p
        self.mix_alpha = mix_alpha

        # Gate regularisation config
        self.use_gate_reg = use_gate_reg
        self.gate_reg_target = gate_reg_target
        self.gate_reg_weight = gate_reg_weight

        # Analysis-only intermediate export.
        # MUST default to False so the training / eval forward pass is byte-for-byte
        # identical to before. Only an offline analysis script flips this on.
        self.return_analysis = return_analysis

        # --- Canonical affine parameters ---
        if affine_type == "group_part":
            self.gamma = nn.Parameter(torch.ones(1, groups, 1, parts_num))
            self.beta = nn.Parameter(torch.zeros(1, groups, 1, parts_num))
        elif affine_type == "group":
            self.gamma = nn.Parameter(torch.ones(1, groups, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, groups, 1, 1))
        elif affine_type == "channel":
            self.gamma = nn.Parameter(torch.ones(1, channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        else:
            raise ValueError(f"Unknown affine_type: {affine_type}")

        # --- Residual gate head: pool->LN->Linear->sigmoid ---
        self.gate_ln = nn.LayerNorm(parts_num)
        self.gate_fc = nn.Linear(parts_num, parts_num)

        nn.init.zeros_(self.gate_fc.weight)
        nn.init.constant_(self.gate_fc.bias, gate_init_bias)

        # Part-wise (Px1) ablation: input-independent learned per-part gate logit.
        # Only created in this mode so default models keep an identical param set.
        if gate_granularity == "part":
            self.gate_part_logit = nn.Parameter(
                torch.full((parts_num,), float(gate_init_bias))
            )

        # Group-part (BxGxP) gate head: sample-adaptive gate that varies per
        # sample, channel-group AND part. The descriptor is pooled from the
        # grouped INPUT feature (z = h_group.mean over Cg), so the gate is always
        # predicted from the input during both training and inference (it is NOT
        # a per-sample learnable parameter). Only created in this mode so other
        # granularities keep an identical parameter set.
        if gate_granularity == "group_part":
            self.gate_gp_ln = nn.LayerNorm([groups, parts_num])
            self.gate_gp_fc = nn.Linear(groups * parts_num, groups * parts_num)
            nn.init.zeros_(self.gate_gp_fc.weight)
            nn.init.constant_(self.gate_gp_fc.bias, gate_init_bias)

        # Diagnostics (populated each forward, no grad)
        self._last_gate_mean: float = 0.0
        self._last_gate_std: float = 0.0
        self._last_delta_norm: float = 0.0
        self._last_var_floor_hit_ratio: float = 0.0

    def _get_g_max(self, iter_num) -> float:
        """Gate cap: ramps from g_max_init to g_max_final over g_max_ramp_iters."""
        if iter_num is None:
            return self.g_max_final
        t = min(1.0, float(iter_num) / max(1, self.g_max_ramp_iters))
        return self.g_max_init + (self.g_max_final - self.g_max_init) * t

    def _compute_stats(self, xg: torch.Tensor):
        """
        Compute per-sample, per-group statistics.
        xg: [B, G, Cg, P]

        SAFETY: NEVER reduce over dim=0 (batch). Only dims 2 and/or 3.

        Returns: (mu, var) with appropriate keepdim shapes.
        """
        if self.stats_axes in ("cg_only_partwise", "cg"):
            # Per-sample, per-group, PER-PART: mean/var over Cg only
            mu = xg.mean(dim=2, keepdim=True)     # [B, G, 1, P]
            var = xg.var(dim=2, unbiased=False, keepdim=True)  # [B, G, 1, P]
        elif self.stats_axes == "cg_p":
            # Legacy: pool over (Cg, P)
            mu = xg.mean(dim=(2, 3), keepdim=True)     # [B, G, 1, 1]
            var = xg.var(dim=(2, 3), unbiased=False, keepdim=True)  # [B, G, 1, 1]
        else:
            raise ValueError(f"Unknown stats_axes: {self.stats_axes}")
        return mu, var

    def _mixstyle(self, xg: torch.Tensor) -> torch.Tensor:
        """
        Stochastic style mixing on grouped tensor (training only).
        xg: [B, G, Cg, P]
        """
        if not self.training or not self.use_mixstyle:
            return xg

        B = xg.size(0)
        if B <= 1:
            return xg
        if torch.rand(1).item() > self.mix_p:
            return xg

        mu, var = self._compute_stats(xg)
        sig = (var.clamp_min(self.var_floor) + self.eps).sqrt()

        perm = torch.randperm(B, device=xg.device)
        ndim = mu.dim()
        a = torch.distributions.Beta(self.mix_alpha, self.mix_alpha).sample(
            (B,) + (1,) * (ndim - 1)
        ).to(xg.device)

        mu_mix = a * mu + (1 - a) * mu[perm]
        sig_mix = a * sig + (1 - a) * sig[perm]

        xg_norm = (xg - mu) / sig
        return xg_norm * sig_mix + mu_mix

    def forward(self, x: torch.Tensor, gate_ctx=None, iter_num=None):
        """
        Args:
            x:        [B, C, P] part-wise embedding (from SeparateFCs)
            gate_ctx: [B, C, P] optional gate context (e.g. BNNeck feature embed_2)
            iter_num: int or None - current training iteration for g_max schedule

        Returns:
            y:         [B, C, P] canonicalized embedding
            pgfc_info: dict with diagnostic scalars
        """
        B, C, P = x.shape
        assert C == self.C, f"Expected C={self.C}, got {C}"
        assert P == self.P, f"Expected P={self.P}, got {P}"

        # 1. Reshape into groups: [B, G, Cg, P]
        xg = x.view(B, self.G, self.Cg, P)

        # 1a. Optional MixStyle (pre-normalization, training only)
        xg = self._mixstyle(xg)

        # 2. Group-instance normalisation (NEVER over batch dim 0)
        mu, var = self._compute_stats(xg)
        var_clamped = var.clamp_min(self.var_floor)
        denom = (var_clamped + self.eps).sqrt()
        xhat = (xg - mu) / denom

        # Track var floor hits
        with torch.no_grad():
            var_floor_hit = (var < self.var_floor).float().mean().item()

        # 3. Canonical affine
        if self.affine_type in ("group_part", "group"):
            xtr_g = self.gamma * xhat + self.beta  # [B, G, Cg, P]
            xtr = xtr_g.view(B, C, P)
        else:  # "channel"
            xtr = xhat.view(B, C, P)
            xtr = self.gamma * xtr + self.beta

        # 4. Gate computation + 5. Gated residual mix
        if self.use_gate:
            if self.gate_granularity == "group_part":
                # Sample-adaptive group-part gate, predicted from the grouped
                # INPUT feature (both train & eval). Varies per (sample, group, part).
                #   descriptor z = h_group.mean(dim=2)            -> [B, G, P]
                #   LayerNorm over [G, P], flatten, Linear(G*P,G*P)
                #   gate = g_max * sigmoid(logits)                -> [B, G, P]
                #   y_group = h_group + gate * (h_tilde - h_group)
                xtr_g_view = xtr.reshape(B, self.G, self.Cg, P)  # h_tilde, [B,G,Cg,P]
                z = xg.mean(dim=2)  # [B, G, P]
                z_ln = self.gate_gp_ln(z)  # LayerNorm over [G, P]
                logits = self.gate_gp_fc(z_ln.reshape(B, self.G * self.P))  # [B, G*P]
                g_max = self._get_g_max(iter_num)
                gate_gp = g_max * torch.sigmoid(logits).view(B, self.G, self.P)  # [B,G,P]

                # Residual correction on the grouped tensor.
                y_group = xg + gate_gp.unsqueeze(2) * (xtr_g_view - xg)  # [B,G,Cg,P]
                y = y_group.view(B, C, P)

                # Diagnostics-compatible part gate (group-averaged) + full gate.
                g_part = gate_gp.mean(dim=1)  # [B, P]
                g = gate_gp  # [B, G, P], used for gate mean/std only
            elif self.gate_granularity == "part":
                # Part-wise (Px1): input-independent learned per-part gate,
                # shared across all samples in the batch.
                g_raw = torch.sigmoid(self.gate_part_logit)  # [P]
                g_raw = g_raw.unsqueeze(0).expand(B, -1)  # [B, P]
            else:
                if self.gate_type == "bnneck":
                    assert gate_ctx is not None, (
                        "gate_type='bnneck' requires gate_ctx (BNNeck feature). "
                        "Pass gate_ctx=embed_2 from SeparateBNNecks output."
                    )
                    gate_input = gate_ctx.detach() if self.stopgrad_gate_ctx else gate_ctx
                else:
                    gate_input = x

                pooled = gate_input.mean(dim=1)  # [B, P]
                logits = self.gate_fc(self.gate_ln(pooled))  # [B, P]
                g_raw = torch.sigmoid(logits)  # [B, P]

                if self.gate_granularity == "sample":
                    # Sample-wise (Bx1): one gate per sample, shared across parts.
                    g_raw = g_raw.mean(dim=1, keepdim=True).expand(-1, self.P)  # [B, P]

            # Gate cap schedule + part-wise residual mix (group_part already
            # produced y / g / g_part above on the grouped tensor).
            if self.gate_granularity != "group_part":
                g_max = self._get_g_max(iter_num)
                g_part = g_raw * g_max  # [B, P]
                g = g_part.unsqueeze(1)  # [B, 1, P]

                # Gated residual mix
                y = x + g * (xtr - x)
        else:
            # Gate disabled: use a CONSTANT residual mix weight gate_const.
            #   gate_const=1.0 => y = xtr (fully corrected; original no-gate behavior)
            #   gate_const=0.5 => y = 0.5*x + 0.5*xtr
            #   gate_const=0.0 => y = x
            g_max = self._get_g_max(iter_num)
            g_part = x.new_full((B, P), float(self.gate_const))  # [B, P]
            g = g_part.unsqueeze(1)  # [B, 1, P]
            y = x + g * (xtr - x)
        y = y.contiguous()

        # 6. Diagnostics (detached, cheap)
        with torch.no_grad():
            gate_mean = g.mean().item()
            gate_std = g.std().item()
            gate_tensor = g_part.detach().clone()  # [B, P]
            gate_part_mean = g_part.mean(dim=0).detach().clone()  # [P]
            gate_part_mean_str = "[" + ", ".join(
                f"{v:.4f}" for v in gate_part_mean.float().cpu().tolist()
            ) + "]"
            delta_norm = (y - x).norm(dim=1).mean().item()
            self._last_gate_mean = gate_mean
            self._last_gate_std = gate_std
            self._last_delta_norm = delta_norm
            self._last_var_floor_hit_ratio = var_floor_hit

        pgfc_info = {
            "pgfc_gate_mean": gate_mean,
            "pgfc_gate_std": gate_std,
            "pgfc_gate": gate_tensor,
            "pgfc_gate_part_mean": gate_part_mean,
            "pgfc_gate_part_mean_str": gate_part_mean_str,
            "gate_for_loss": g_part,
            "gate": gate_tensor,
            "gate_part_mean": gate_part_mean,
            "pgfc_delta_norm": delta_norm,
            "pgfc_var_floor_hit_ratio": var_floor_hit,
            "pgfc_g_max": g_max,
        }

        # Full sample-adaptive group-part gate [B,G,P] (group_part mode only).
        if self.gate_granularity == "group_part":
            pgfc_info["pgfc_gate_group_part"] = g.detach()

        # ----------------------------------------------------------------------
        # Analysis-only intermediate export (group-wise sensitivity analysis).
        #
        # Disabled by default (self.return_analysis is False), so this block is a
        # complete no-op for training and inference and cannot affect the loss or
        # retrieval features in any way. When an offline analysis script sets
        # `return_analysis=True`, the per-stage feature tensors are attached
        # (detached, no grad) to pgfc_info under the "analysis" key so they can be
        # collected by tools/analysis/analyze_pgfc_groups.py.
        #
        # Shapes (with B=batch, C=channels, G=groups, Cg=C//G, P=parts):
        #   X            [B, C, P]      part-wise feature BEFORE PGFC correction
        #   X_grp        [B, G, Cg, P]  reshaped group feature BEFORE normalization
        #   X_tilde_grp  [B, G, Cg, P]  group-part normalized + affine feature
        #   X_tilde      [B, C, P]      corrected feature after reshape
        #   M            [B, P]         part-wise gate / residual-mix weight
        #   Y            [B, C, P]      final corrected (gated) feature
        # ----------------------------------------------------------------------
        if self.return_analysis:
            with torch.no_grad():
                if self.affine_type in ("group_part", "group"):
                    x_tilde_grp = xtr_g
                else:  # "channel": rebuild the grouped view for analysis
                    x_tilde_grp = xtr.view(B, self.G, self.Cg, P)
                pgfc_info["analysis"] = {
                    "X": x.detach(),
                    "X_grp": xg.detach(),
                    "X_tilde_grp": x_tilde_grp.detach(),
                    "X_tilde": xtr.detach(),
                    "M": g_part.detach(),
                    "Y": y.detach(),
                }

        # Guardrail: gate saturation warning (only meaningful when gate is active)
        if (self.use_gate and self.training and iter_num is not None
                and iter_num < 2000 and gate_mean > 0.7):
            print(
                f"[PGFC WARNING] iter={iter_num}: gate_mean={gate_mean:.3f} > 0.7 "
                f"early in training. Consider lowering g_max_init or making "
                f"gate_init_bias more negative."
            )

        return y, pgfc_info

    def gate_reg_loss(self) -> torch.Tensor:
        """
        Tiny penalty to keep gate from collapsing to 0 or 1.
        Returns scalar tensor (0 if disabled).
        """
        if not self.use_gate_reg:
            return torch.tensor(0.0, device=self.gamma.device)
        diff = self._last_gate_mean - self.gate_reg_target
        return torch.tensor(self.gate_reg_weight * (diff ** 2), device=self.gamma.device)

    @property
    def diagnostics(self) -> dict:
        """Return latest diagnostic scalars (for logging)."""
        return {
            "pgfc_gate_mean": self._last_gate_mean,
            "pgfc_gate_std": self._last_gate_std,
            "pgfc_delta_norm": self._last_delta_norm,
            "pgfc_var_floor_hit_ratio": self._last_var_floor_hit_ratio,
        }
