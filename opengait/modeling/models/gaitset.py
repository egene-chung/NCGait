import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import (SeparateFCs, SeparateBNNecks, BasicConv2d,
                       SetBlockWrapper, HorizontalPoolingPyramid,
                       PackSequenceWrapper)
from ..pgfc import GatedPGFC


class PartwiseBNNeck(nn.Module):
    def __init__(self, channels, parts_num):
        super(PartwiseBNNeck, self).__init__()
        self.channels = channels
        self.parts_num = parts_num
        self.bn = nn.BatchNorm1d(channels * parts_num)

    def forward(self, x):
        n, c, p = x.size()
        if c != self.channels or p != self.parts_num:
            raise ValueError(
                f"Expected [N, {self.channels}, {self.parts_num}], got {list(x.shape)}")
        x = self.bn(x.view(n, c * p)).view(n, c, p)
        return F.normalize(x, p=2, dim=1)


class GaitSet(BaseModel):
    """
        GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition
    """

    def build_network(self, model_cfg):
        in_c = model_cfg['in_channels']
        self.set_block1 = nn.Sequential(BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2 = nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block3 = nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))

        self.gl_block2 = copy.deepcopy(self.set_block2)
        self.gl_block3 = copy.deepcopy(self.set_block3)

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.set_pooling = PackSequenceWrapper(torch.max)

        head_cfg = model_cfg['SeparateFCs']
        self.Head = SeparateFCs(**head_cfg)

        # GaitSet is triplet-only by default.  Build the standard OpenGait
        # BNNeck/classifier when a config requests an auxiliary CE objective.
        bnneck_cfg = model_cfg.get('SeparateBNNecks')
        self.BNNecks = SeparateBNNecks(**bnneck_cfg) \
            if bnneck_cfg is not None else None

        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        # --- PGFC (optional) ---
        # Triplet-only configs have no classifier BNNeck, so synthesize a
        # lightweight fallback when PGFC requests a BNNeck gate context.
        self.use_pgfc = model_cfg.get('use_pgfc', False)
        if self.use_pgfc:
            pgfc_cfg = model_cfg.get('pgfc', {})
            fc_out_channels = head_cfg['out_channels']
            parts_num = head_cfg['parts_num']
            gate_type = pgfc_cfg.get('gate_type', 'part')
            # Prefer the classifier BNNeck as PGFC's gate context.  Keep the
            # lightweight fallback for existing triplet-only PGFC configs.
            self.PGFCGateBN = PartwiseBNNeck(
                fc_out_channels, parts_num) \
                if gate_type == 'bnneck' and self.BNNecks is None else None
            self.pgfc = GatedPGFC(
                channels=fc_out_channels,
                groups=pgfc_cfg.get('groups', 16),
                eps=pgfc_cfg.get('eps', 1e-4),
                var_floor=pgfc_cfg.get('var_floor', 1e-4),
                parts_num=parts_num,
                gate_init_bias=pgfc_cfg.get('gate_init_bias', -4.0),
                gate_type=gate_type,
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
            from utils import get_msg_mgr
            get_msg_mgr().log_info(
                f"[GaitSet] PGFC enabled: C={fc_out_channels}, "
                f"G={pgfc_cfg.get('groups', 16)}, "
                f"gate={gate_type}")
        else:
            self.pgfc = None
            self.PGFCGateBN = None

        eval_cfg = getattr(self, 'cfgs', {}).get('evaluator_cfg', {}) if hasattr(self, 'cfgs') else {}
        self.use_pgfc_eval_fusion = eval_cfg.get('use_pgfc_fusion', False)

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

    def _should_log_pgfc_part(self, iter_num):
        if not self.training or not self._is_rank0():
            return False
        log_iter = getattr(getattr(self, 'msg_mgr', None), 'log_iter', None)
        if isinstance(log_iter, int) and log_iter > 0 and iter_num % log_iter == 0:
            return True
        return iter_num % 500 == 0

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]  # [n, s, h, w] or [n, s, c, h, w]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        elif len(sils.size()) == 5:
            sils = sils.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                f"Expected silhouette input with 4 or 5 dims, got {list(sils.shape)}")

        del ipts
        outs = self.set_block1(sils)
        gl = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl = gl + self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = gl + outs

        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p]
        feature2 = self.HPP(gl)  # [n, c, p]
        feature = torch.cat([feature1, feature2], -1)  # [n, c, p]
        embs = self.Head(feature)  # [n, c, p]

        if self.BNNecks is not None:
            bn_embs, logits = self.BNNecks(embs)
        else:
            bn_embs, logits = None, None

        pgfc_info = None
        pgfc_input = None
        if self.use_pgfc and self.pgfc is not None:
            iter_num = getattr(self, 'iteration', None)
            pgfc_input = embs
            if self.pgfc.gate_type == 'bnneck':
                gate_ctx = bn_embs if bn_embs is not None \
                    else self.PGFCGateBN(embs)
            else:
                gate_ctx = None
            embs_for_loss, pgfc_info = self.pgfc(
                pgfc_input, gate_ctx=gate_ctx, iter_num=iter_num)
            assert embs_for_loss.shape == embs.shape, \
                f"PGFC shape mismatch: {embs_for_loss.shape} vs {embs.shape}"
        else:
            embs_for_loss = embs

        if self.use_pgfc and self.pgfc is not None:
            if not self.training and self.use_pgfc_eval_fusion:
                embed = torch.cat([pgfc_input, embs_for_loss], dim=1)
            else:
                embed = embs_for_loss
        else:
            embed = embs_for_loss

        n, _, s, h, w = sils.size()
        training_feat = {
            'triplet': {'embeddings': embs_for_loss, 'labels': labs}
        }
        if logits is not None:
            training_feat['softmax'] = {'logits': logits, 'labels': labs}

        retval = {
            'training_feat': training_feat,
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        if pgfc_info is not None:
            retval['visual_summary']['scalar/pgfc_gate_mean'] = pgfc_info['pgfc_gate_mean']
            retval['visual_summary']['scalar/pgfc_gate_std'] = pgfc_info['pgfc_gate_std']
            retval['visual_summary']['scalar/pgfc_delta_norm'] = pgfc_info['pgfc_delta_norm']
            retval['visual_summary']['scalar/pgfc_g_max'] = pgfc_info.get('pgfc_g_max', 1.0)

            next_iter = getattr(self, 'iteration', 0) + 1
            if self._should_log_pgfc_part(next_iter):
                self._add_part_scalars(
                    retval['visual_summary'], 'pgfc_gate',
                    pgfc_info.get('pgfc_gate_part_mean', None))
                delta_norm_part = self._compute_delta_norm_part(
                    pgfc_input, embs_for_loss)
                self._add_part_scalars(
                    retval['visual_summary'], 'pgfc_delta_norm', delta_norm_part)

            retval['pgfc_info'] = pgfc_info

        return retval
