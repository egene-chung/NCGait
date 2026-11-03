import torch

from ..base_model import BaseModel
from ..modules import (
    SetBlockWrapper,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateFCs,
    SeparateBNNecks,
)
from ..pgfc import GatedPGFC

from einops import rearrange


class Baseline(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.use_pgfc = model_cfg.get('use_pgfc', False)
        if self.use_pgfc:
            pgfc_cfg = model_cfg.get('pgfc', {})
            fc_out_channels = model_cfg['SeparateFCs']['out_channels']
            parts_num = model_cfg['bin_num'][0] if isinstance(model_cfg['bin_num'], list) else model_cfg['bin_num']
            self.pgfc = GatedPGFC(
                channels=fc_out_channels,
                groups=pgfc_cfg.get('groups', 16),
                eps=pgfc_cfg.get('eps', 1e-4),
                var_floor=pgfc_cfg.get('var_floor', 1e-4),
                parts_num=parts_num,
                gate_init_bias=pgfc_cfg.get('gate_init_bias', -4.0),
                gate_type=pgfc_cfg.get('gate_type', 'bnneck'),
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

        eval_cfg = getattr(self, 'cfgs', {}).get('evaluator_cfg', {}) if hasattr(self, 'cfgs') else {}
        self.use_pgfc_eval_fusion = eval_cfg.get('use_pgfc_fusion', False)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)
        outs = self.TP(outs, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs)

        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)

        pgfc_info = None
        if self.use_pgfc and self.pgfc is not None:
            iter_num = getattr(self, 'iteration', None)
            gate_ctx = embed_2 if self.pgfc.gate_type == "bnneck" else None
            embed_1_metric, pgfc_info = self.pgfc(embed_1, gate_ctx=gate_ctx, iter_num=iter_num)
            assert embed_1_metric.shape == embed_1.shape, \
                f"PGFC shape mismatch: {embed_1_metric.shape} vs {embed_1.shape}"
            embed_1_for_loss = embed_1_metric
            if not self.training and self.use_pgfc_eval_fusion:
                embed = torch.cat([embed_1, embed_1_metric], dim=1)
            else:
                embed = embed_1_metric
        else:
            embed_1_for_loss = embed_1
            embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1_for_loss, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')
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
            gate_part_mean = pgfc_info.get('pgfc_gate_part_mean', None)
            if gate_part_mean is not None:
                for part_idx, part_gate in enumerate(gate_part_mean):
                    retval['visual_summary'][f'scalar/pgfc_gate_part/part_{part_idx:02d}'] = part_gate
                retval['visual_summary']['histogram/pgfc_gate_part_mean'] = gate_part_mean
                retval['visual_summary']['text/pgfc_gate_part_mean_vector'] = \
                    pgfc_info.get('pgfc_gate_part_mean_str', str(gate_part_mean.detach().cpu().tolist()))
            retval['pgfc_info'] = pgfc_info

        return retval
