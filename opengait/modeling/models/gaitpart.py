import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks
from ..pgfc import GatedPGFC
from utils import clones


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret


class TemporalFeatureAggregator(nn.Module):
    def __init__(self, in_channels, squeeze=4, parts_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        # MTB1
        conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        # MTB1
        conv3x3 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

        # Temporal Pooling, TP
        self.TP = torch.max

    def forward(self, x):
        """
          Input:  x,   [n, c, s, p]
          Output: ret, [n, c, p]
        """
        n, c, s, p = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()  # [p, n, c, s]
        feature = x.split(1, 0)  # [[1, n, c, s], ...]
        x = x.view(-1, c, s)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = self.TP(feature3x1 + feature3x3, dim=-1)[0]  # [p, n, c]
        ret = ret.permute(1, 2, 0).contiguous()  # [n, p, c]
        return ret


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


class GaitPart(BaseModel):
    def __init__(self, *args, **kargs):
        super(GaitPart, self).__init__(*args, **kargs)
        """
            GaitPart: Temporal Part-based Model for Gait Recognition
        """

    def build_network(self, model_cfg):

        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        head_cfg = model_cfg['SeparateFCs']
        self.Head = SeparateFCs(**model_cfg['SeparateFCs'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.HPP = SetBlockWrapper(
            HorizontalPoolingPyramid(bin_num=model_cfg['bin_num']))
        self.TFA = PackSequenceWrapper(TemporalFeatureAggregator(
            in_channels=head_cfg['in_channels'], parts_num=head_cfg['parts_num']))

        # GaitPart is triplet-only by default.  Build the standard OpenGait
        # BNNeck/classifier when a config requests an auxiliary CE objective.
        bnneck_cfg = model_cfg.get('SeparateBNNecks')
        self.BNNecks = SeparateBNNecks(**bnneck_cfg) \
            if bnneck_cfg is not None else None

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
            self.PGFCGateBN = None

        eval_cfg = getattr(self, 'cfgs', {}).get('evaluator_cfg', {}) if hasattr(self, 'cfgs') else {}
        self.use_pgfc_eval_fusion = eval_cfg.get('use_pgfc_fusion', False)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        out = self.Backbone(sils)  # [n, c, s, h, w]
        out = self.HPP(out)  # [n, c, s, p]
        out = self.TFA(out, seqL)  # [n, c, p]

        embs = self.Head(out)  # [n, c, p]

        if self.BNNecks is not None:
            bn_embs, logits = self.BNNecks(embs)
        else:
            bn_embs, logits = None, None

        pgfc_info = None
        if self.use_pgfc and self.pgfc is not None:
            iter_num = getattr(self, 'iteration', None)
            if self.pgfc.gate_type == 'bnneck':
                gate_ctx = bn_embs if bn_embs is not None \
                    else self.PGFCGateBN(embs)
            else:
                gate_ctx = None
            embs_metric, pgfc_info = self.pgfc(
                embs, gate_ctx=gate_ctx, iter_num=iter_num)
            if not self.training and self.use_pgfc_eval_fusion:
                inference_embs = torch.cat([embs, embs_metric], dim=1)
            else:
                inference_embs = embs_metric
            training_embs = embs_metric
        else:
            inference_embs = embs
            training_embs = embs

        n, _, s, h, w = sils.size()
        training_feat = {
            'triplet': {'embeddings': training_embs, 'labels': labs}
        }
        if logits is not None:
            training_feat['softmax'] = {'logits': logits, 'labels': labs}

        retval = {
            'training_feat': training_feat,
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': inference_embs
            }
        }
        if pgfc_info is not None:
            retval['visual_summary']['scalar/pgfc_gate_mean'] = pgfc_info['pgfc_gate_mean']
            retval['visual_summary']['scalar/pgfc_gate_std'] = pgfc_info['pgfc_gate_std']
            retval['visual_summary']['scalar/pgfc_delta_norm'] = pgfc_info['pgfc_delta_norm']
            retval['visual_summary']['scalar/pgfc_g_max'] = pgfc_info.get('pgfc_g_max', 1.0)
            retval['pgfc_info'] = pgfc_info
        return retval
