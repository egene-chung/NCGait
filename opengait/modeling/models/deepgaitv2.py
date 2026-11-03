import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import (
    SetBlockWrapper,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateFCs,
    SeparateBNNecks,
    conv1x1,
    conv3x3,
    BasicBlock2D,
    BasicBlockP3D,
    BasicBlock3D,
)
from ..pgfc import GatedPGFC, BatchNormCanon, PartGateOnly

from einops import rearrange


blocks_map = {
    '2d': BasicBlock2D,
    'p3d': BasicBlockP3D,
    '3d': BasicBlock3D
}


class DeepGaitV2(BaseModel):

    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers = model_cfg['Backbone']['layers']
        channels = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        self.normalize_before_bn = model_cfg.get('normalize_before_bn', False)
        self.triplet_normalize = model_cfg.get('triplet_normalize', self.normalize_before_bn)

        if mode == '3d':
            strides = [
                [1, 1],
                [1, 2, 2],
                [1, 2, 2],
                [1, 1, 1]
            ]
        else:
            strides = [
                [1, 1],
                [2, 2],
                [2, 2],
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(self.make_layer(
            BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))
        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d':
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        self.use_pgfc = model_cfg.get('use_pgfc', False)
        if self.use_pgfc:
            pgfc_cfg = model_cfg.get('pgfc', {})
            fc_out_channels = channels[2]
            self.pgfc = GatedPGFC(
                channels=fc_out_channels,
                groups=pgfc_cfg.get('groups', 16),
                eps=pgfc_cfg.get('eps', 1e-4),
                var_floor=pgfc_cfg.get('var_floor', 1e-4),
                parts_num=16,
                gate_init_bias=pgfc_cfg.get('gate_init_bias', -4.0),
                gate_type=pgfc_cfg.get('gate_type', 'bnneck'),
                gate_granularity=pgfc_cfg.get('gate_granularity', 'sample_part'),
                use_gate=pgfc_cfg.get('use_gate', True),
                gate_const=pgfc_cfg.get('gate_const', 1.0),
                stopgrad_gate_ctx=pgfc_cfg.get('stopgrad_gate_ctx', True),
                affine_type=pgfc_cfg.get('affine_type', 'group_part'),
                stats_axes=pgfc_cfg.get('stats_axes', 'cg_only_partwise'),
                g_max_init=pgfc_cfg.get('g_max_init', 0.1),
                g_max_final=pgfc_cfg.get('g_max_final', 1.0),
                g_max_ramp_iters=pgfc_cfg.get('g_max_ramp_iters', 5000),
                use_mixstyle=pgfc_cfg.get('use_mixstyle', False),
                mix_p=pgfc_cfg.get('mix_p', 0.5),
                mix_alpha=pgfc_cfg.get('mix_alpha', 0.1),
                use_gate_reg=pgfc_cfg.get('use_gate_reg', False),
                gate_reg_target=pgfc_cfg.get('gate_reg_target', 0.2),
                gate_reg_weight=pgfc_cfg.get('gate_reg_weight', 1e-4),
            )
        else:
            self.pgfc = None

        # Plain BatchNorm canonicalizer (alternative to PGFC): batch-norm the
        # [B, C, P] embedding like the BNNeck does before CE, and use it as the
        # loss / retrieval feature. Mutually exclusive with use_pgfc.
        self.use_bn_canon = model_cfg.get('use_bn_canon', False)
        if self.use_bn_canon:
            bn_canon_cfg = model_cfg.get('bn_canon', {})
            self.bn_canon = BatchNormCanon(
                channels=channels[2],
                parts_num=bn_canon_cfg.get('parts_num', 16),
                parallel_BN1d=bn_canon_cfg.get('parallel_BN1d', True),
            )
        else:
            self.bn_canon = None

        # Part-wise Gate ONLY ablation (alternative to PGFC / bn_canon): learn a
        # sample-adaptive per-part residual scale and apply it to the ORIGINAL
        # feature. No group-wise normalization, no corrected feature h_tilde.
        # Mutually exclusive with use_pgfc / use_bn_canon.
        self.use_part_gate = model_cfg.get('use_part_gate', False)
        if self.use_part_gate:
            part_gate_cfg = model_cfg.get('part_gate', {})
            self.part_gate = PartGateOnly(
                channels=channels[2],
                parts_num=part_gate_cfg.get('parts_num', 16),
                alpha=part_gate_cfg.get('alpha', 0.5),
            )
        else:
            self.part_gate = None

        eval_cfg = getattr(self, 'cfgs', {}).get('evaluator_cfg', {}) if hasattr(self, 'cfgs') else {}
        self.use_pgfc_eval_fusion = eval_cfg.get('use_pgfc_fusion', False)
        # Eval-only: bypass PGFC correction and retrieve with the normalized raw
        # feature (pgfc_input) instead of the gated/canonicalized output.
        self.use_pgfc_eval_raw = eval_cfg.get('use_pgfc_raw', False)

    def set_pgfc_return_analysis(self, flag: bool = True):
        """Analysis-only helper.

        Toggle PGFC intermediate-feature export on/off. This only flips a flag on
        the (already-built) GatedPGFC submodule and has NO effect unless the PGFC
        block is present. It is intended to be called by the offline analysis
        script (tools/analysis/analyze_pgfc_groups.py) AFTER loading a checkpoint;
        normal training / testing never calls it, so default behaviour is
        unchanged. Returns True if the flag was actually applied.
        """
        if getattr(self, 'use_pgfc', False) and getattr(self, 'pgfc', None) is not None:
            self.pgfc.return_analysis = bool(flag)
            return True
        return False

    @staticmethod
    def _format_part_vector(vec):
        vec = vec.detach().float().cpu().tolist()
        return "[" + ", ".join(f"{v:.4f}" for v in vec) + "]"

    @staticmethod
    def _is_rank0():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    @staticmethod
    def _add_part_scalars(visual_summary, name, values):
        if values is None:
            return
        values = values.detach()
        for part_idx, value in enumerate(values):
            visual_summary[f"scalar/{name}_part_{part_idx:02d}"] = value.detach()

    @staticmethod
    def _compute_delta_norm_part(raw_feat, pgfc_feat):
        if raw_feat is None or raw_feat.shape != pgfc_feat.shape:
            return None
        with torch.no_grad():
            return (pgfc_feat.detach() - raw_feat.detach()).norm(dim=1).mean(dim=0).detach()

    @staticmethod
    def _part_pair_distance(feat):
        feat = feat.detach().float().permute(2, 0, 1).contiguous()  # [P, B, C]
        squared = torch.sum(feat ** 2, dim=-1)
        dist = squared.unsqueeze(2) + squared.unsqueeze(1) - 2 * feat.matmul(feat.transpose(1, 2))
        return torch.sqrt(F.relu(dist))

    @staticmethod
    def _hard_margin_part(dist, pos_mask, neg_mask, valid_anchor):
        pos_dist = dist.masked_fill(~pos_mask.unsqueeze(0), -float("inf")).amax(dim=2)
        neg_dist = dist.masked_fill(~neg_mask.unsqueeze(0), float("inf")).amin(dim=2)
        return (neg_dist - pos_dist)[:, valid_anchor].mean(dim=1).detach()

    def _compute_margin_gain_part(self, raw_feat, pgfc_feat, labels):
        with torch.no_grad():
            if raw_feat is None or raw_feat.shape != pgfc_feat.shape:
                return None
            if raw_feat.ndim != 3 or raw_feat.size(0) <= 1:
                return None

            batch_size = raw_feat.size(0)
            labels = labels.detach().to(device=raw_feat.device).view(-1)
            if labels.numel() != batch_size:
                return None

            same_pid = labels.unsqueeze(0) == labels.unsqueeze(1)
            not_self = ~torch.eye(batch_size, dtype=torch.bool, device=raw_feat.device)
            pos_mask = same_pid & not_self
            neg_mask = ~same_pid
            valid_anchor = pos_mask.any(dim=1) & neg_mask.any(dim=1)
            if not torch.any(valid_anchor):
                return None

            raw_margin = self._hard_margin_part(
                self._part_pair_distance(raw_feat), pos_mask, neg_mask, valid_anchor)
            pgfc_margin = self._hard_margin_part(
                self._part_pair_distance(pgfc_feat), pos_mask, neg_mask, valid_anchor)
            return (pgfc_margin - raw_margin).detach()

    def _should_log_pgfc_part(self, iter_num):
        if not self.training or not self._is_rank0():
            return False
        log_iter = getattr(getattr(self, 'msg_mgr', None), 'log_iter', None)
        if isinstance(log_iter, int) and log_iter > 0 and iter_num % log_iter == 0:
            return True
        return iter_num % 500 == 0

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for _ in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, stride=s))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        outs = self.TP(out4, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs)
        embed_1_raw = self.FCs(feat)

        if self.normalize_before_bn:
            embed_1_bn = F.normalize(embed_1_raw, p=2, dim=1)
        else:
            embed_1_bn = embed_1_raw

        embed_2, logits = self.BNNecks(embed_1_bn)

        pgfc_info = None
        pgfc_input = None
        if self.use_pgfc and self.pgfc is not None:
            iter_num = getattr(self, 'iteration', None)
            gate_ctx = embed_2 if self.pgfc.gate_type == "bnneck" else None
            pgfc_input = F.normalize(embed_1_raw, p=2, dim=1) if self.triplet_normalize else embed_1_raw
            embed_1_for_loss, pgfc_info = self.pgfc(pgfc_input, gate_ctx=gate_ctx, iter_num=iter_num)
            assert embed_1_for_loss.shape == embed_1_raw.shape, \
                f"PGFC shape mismatch: {embed_1_for_loss.shape} vs {embed_1_raw.shape}"
        elif self.use_part_gate and self.part_gate is not None:
            iter_num = getattr(self, 'iteration', None)
            pgfc_input = F.normalize(embed_1_raw, p=2, dim=1) if self.triplet_normalize else embed_1_raw
            embed_1_for_loss, pgfc_info = self.part_gate(pgfc_input, iter_num=iter_num)
            assert embed_1_for_loss.shape == embed_1_raw.shape, \
                f"PartGate shape mismatch: {embed_1_for_loss.shape} vs {embed_1_raw.shape}"
        elif self.use_bn_canon and self.bn_canon is not None:
            bn_canon_input = F.normalize(embed_1_raw, p=2, dim=1) if self.triplet_normalize else embed_1_raw
            embed_1_for_loss = self.bn_canon(bn_canon_input)
        else:
            embed_1_for_loss = F.normalize(embed_1_raw, p=2, dim=1) if self.triplet_normalize else embed_1_raw

        if self.inference_use_emb2:
            embed = embed_2
        elif self.use_part_gate and self.part_gate is not None:
            embed = embed_1_for_loss
        elif self.use_bn_canon and self.bn_canon is not None:
            embed = embed_1_for_loss
        elif self.use_pgfc and self.pgfc is not None:
            if not self.training and self.use_pgfc_eval_fusion:
                pgfc_input = F.normalize(embed_1_raw, p=2, dim=1) if self.triplet_normalize else embed_1_raw
                embed = torch.cat([pgfc_input, embed_1_for_loss], dim=1)
            elif not self.training and self.use_pgfc_eval_raw:
                # Normalized raw feature only (skip PGFC correction).
                embed = pgfc_input
            else:
                embed = embed_1_for_loss
        else:
            embed = embed_1_bn

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1_for_loss, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        if pgfc_info is not None:
            next_iter = getattr(self, 'iteration', 0) + 1
            if self._should_log_pgfc_part(next_iter):
                self._add_part_scalars(
                    retval['visual_summary'], 'pgfc_gate', pgfc_info.get('pgfc_gate_part_mean', None))
                delta_norm_part = self._compute_delta_norm_part(
                    pgfc_input, embed_1_for_loss)
                self._add_part_scalars(
                    retval['visual_summary'], 'pgfc_delta_norm', delta_norm_part)

                margin_gain_part = self._compute_margin_gain_part(
                    pgfc_input, embed_1_for_loss, labs)
                self._add_part_scalars(
                    retval['visual_summary'], 'margin_gain', margin_gain_part)

            retval['pgfc_info'] = pgfc_info

        return retval
