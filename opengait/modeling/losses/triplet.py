import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0,
                 neg_mining="standard",
                 vm_k_neg=4,
                 vm_aux_weight=0.2,
                 vm_primary_max_priority=2,
                 vm_use_viewset=True,
                 vm_global_topk=False,
                 vm_use_all_unmatched=False,
                 vm_strict_view_weighting=False):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.neg_mining = neg_mining
        self.vm_k_neg = vm_k_neg
        self.vm_aux_weight = vm_aux_weight
        self.vm_primary_max_priority = vm_primary_max_priority
        self.vm_use_viewset = vm_use_viewset
        # When True the full weight is reserved for negatives that really share
        # the anchor's view label; anything else (including the global hard
        # negative used when no view-matched negative exists) gets
        # ``vm_aux_weight``. Default False keeps the priority-based weighting.
        self.vm_strict_view_weighting = vm_strict_view_weighting
        # When True, ignore view-based priority and pick the globally closest
        # (hardest) negatives across ALL views (top-1 with vm_k_neg=1).
        self.vm_global_topk = vm_global_topk
        # Used only by neg_mining == "view_unmatched": when True, use ALL
        # cross-view (different-view) negatives instead of the top-k hardest.
        self.vm_use_all_unmatched = vm_use_all_unmatched
        self._vm_log_counter = 0

        if self.neg_mining == "strict_view_match_top1" and self.vm_k_neg != 1:
            raise ValueError(
                "strict_view_match_top1 requires vm_k_neg=1, got "
                f"{self.vm_k_neg}")

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels, view_id=None, view_set_id=None,
                outfit_id=None):
        # embeddings: [n, c, p], label: [n]
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]

        if self.neg_mining == "strict_view_match_top1" and view_id is not None:
            loss_avg, hard_loss, loss_num, total_triplets = \
                self._strict_view_match_top1_triplet(
                    dist, labels, view_id)
        elif self.neg_mining == "view_match_negative_mining" and view_id is not None:
            loss_avg, hard_loss, loss_num, total_triplets = self._view_match_negative_mining_triplet(
                dist, labels, view_id, view_set_id, outfit_id)
        elif self.neg_mining == "view_unmatched" and view_id is not None:
            loss_avg, hard_loss, loss_num, total_triplets = self._view_unmatched_triplet(
                dist, labels, view_id, outfit_id)
        else:
            ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
            dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
            loss = F.relu(dist_diff + self.margin)

            hard_loss = torch.max(loss, -1)[0]
            loss_avg, loss_num = self.AvgNonZeroReducer(loss)
            total_triplets = torch.tensor(float(dist_diff.shape[-1]), device=dist.device).expand(dist.size(0))

        active_ratio = loss_num / (total_triplets + 1e-9)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'total_triplets': total_triplets.detach().clone(),
            'active_ratio': active_ratio.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def _strict_view_match_top1_triplet(self, dist, labels, view_id):
        """Rank-independent, strict same-view top-1 negative mining.

        Candidate negatives satisfy both ``label_j != label_i`` and
        ``view_j == view_i``.  Hidden/invalid view labels never match.  The
        closest candidate is selected with the distance averaged over parts;
        its index is then shared by every part.  There is deliberately no
        cross-view or same-rank fallback.
        """
        p, n, _ = dist.size()

        matches = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
        neg_mask = ~matches
        ap_dist = dist[:, matches].view(p, n, -1, 1)

        valid_view = view_id >= 0
        same_view = ((view_id.unsqueeze(1) == view_id.unsqueeze(0))
                     & valid_view.unsqueeze(1) & valid_view.unsqueeze(0))
        candidate_mask = neg_mask & same_view
        has_candidate = candidate_mask.any(dim=1)

        # Mining is sample-level: average part distances, select one batch
        # index per anchor, then reuse that index for every body part.
        dist_mean = dist.mean(dim=0)
        score = dist_mean.masked_fill(~candidate_mask, float("inf"))
        idx_neg = score.argmin(dim=1)                              # [n]
        idx_neg_exp = idx_neg.view(1, n, 1).expand(p, -1, -1)
        an_dist = dist.gather(dim=2, index=idx_neg_exp).unsqueeze(2)

        raw = F.relu(ap_dist - an_dist + self.margin)             # [p,n,K_pos,1]
        weight = has_candidate.float().view(1, n, 1, 1)
        weighted_raw = raw * weight
        raw_flat = weighted_raw.reshape(p, -1)
        weight_flat = weight.expand_as(raw).contiguous().view(p, -1)
        active = (raw != 0).float() * weight

        loss_sum = raw_flat.sum(dim=-1)
        active_count = active.reshape(p, -1).sum(dim=-1)
        loss_avg = loss_sum / (active_count + 1e-9)
        loss_avg[active_count == 0] = 0

        hard_loss = raw_flat.max(dim=-1).values
        loss_num = active_count
        total_triplets = weight_flat.sum(dim=-1)

        with torch.no_grad():
            any_selected = has_candidate.any()
            selected_ratio = has_candidate.float().mean()
            zero = dist.new_tensor(0.0)
            one = dist.new_tensor(1.0)
            self.info['vm/view_matched_neg_ratio'] = one if any_selected else zero
            self.info['vm/global_fallback_neg_ratio'] = zero
            self.info['vm/aux_weighted_neg_ratio'] = zero
            self.info['vm/anchors_without_view_matched_neg_ratio'] = \
                1.0 - selected_ratio
            self.info['vm/valid_view_anchor_ratio'] = valid_view.float().mean()
            self.info['vm/invalid_view_ratio'] = (~valid_view).float().mean()
            self.info['vm/selected_neg_same_view_ratio'] = \
                one if any_selected else zero
            self.info['vm/selected_neg_cross_view_ratio'] = zero

            selected_dist = dist_mean.gather(1, idx_neg.unsqueeze(1)).squeeze(1)
            selected_dist = selected_dist[has_candidate]
            if selected_dist.numel() > 0:
                self.info['vm/selected_neg_dist_mean'] = selected_dist.mean()
                self.info['vm/selected_neg_dist_std'] = \
                    selected_dist.std(unbiased=False)

        return loss_avg, hard_loss, loss_num, total_triplets

    def _view_match_negative_mining_triplet(self, dist, labels, view_id, view_set_id,
                              outfit_id=None):
        """View match negative mining with priority-based top-k selection.

        Positive pairs are unchanged from the original Convert2Triplets logic.
        Negatives are selected via a fixed-k top-k scheme where view-match
        negatives (same view-set / same view) are prioritised, unless
        ``vm_global_topk`` is set (then the globally closest negatives are
        selected regardless of view).
        """
        p, n, _ = dist.size()

        # -- positive: identical to original --
        matches = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()  # [n, n]
        neg_mask = ~matches  # [n, n]
        ap_dist = dist[:, matches].view(p, n, -1, 1)  # [p, n, K_pos, 1]

        # -- negative candidate count safety --
        # Sizing k on the max (not the min) keeps anchors that do have negatives
        # productive; anchors with fewer than k are zero-weighted further down.
        # Returning zero here means the batch holds a single identity, which is
        # a different situation from "no view-matched negative exists" -- that
        # case falls through to the aux-weighted global hard negative.
        k_neg = min(self.vm_k_neg, int(neg_mask.sum(dim=1).max().item()))
        if k_neg == 0:
            zero = dist.new_zeros(p)
            return zero, zero, zero, zero

        # -- priority map (lower = preferred) --
        # A negative view index marks a hidden GT view (Partial-GT experiments).
        # Two such samples share nothing, so they must not be called same-view;
        # their anchors fall through to the global-negative path below.
        valid_view = view_id >= 0                                   # [n]
        same_view = ((view_id.unsqueeze(1) == view_id.unsqueeze(0))
                     & valid_view.unsqueeze(1) & valid_view.unsqueeze(0))  # [n, n]
        if self.vm_use_viewset and view_set_id is not None:
            same_vs = (view_set_id.unsqueeze(1) == view_set_id.unsqueeze(0))
        else:
            same_vs = torch.zeros_like(same_view)

        prio = torch.full((n, n), 3.0, device=dist.device)
        prio[same_view] = 2.0
        prio[same_vs] = 1.0
        prio[same_vs & same_view] = 0.0

        # -- composite score --
        dist_mean = dist.mean(dim=0)  # [n, n]
        if self.vm_global_topk:
            # ignore view priority: select the globally closest (hardest)
            # negatives across ALL views, distance-only.
            score = dist_mean
        else:
            # priority-first, distance-second
            neg_dists = dist_mean[neg_mask]
            big = (neg_dists.max().detach() + 1.0) if neg_dists.numel() > 0 else 1.0
            score = prio * big + dist_mean
        score = score.masked_fill(~neg_mask, float("inf"))

        # -- top-k selection (smallest score = highest priority, then closest) --
        _, idx_neg = torch.topk(score, k=k_neg, dim=1, largest=False)  # [n, k_neg]

        # -- gather an_dist from all parts --
        idx_neg_exp = idx_neg.unsqueeze(0).expand(p, -1, -1)  # [p, n, k_neg]
        an_dist = dist.gather(dim=2, index=idx_neg_exp)       # [p, n, k_neg]
        an_dist = an_dist.unsqueeze(2)                         # [p, n, 1, k_neg]

        # -- per-negative weight --
        prio_sel = prio.gather(dim=1, index=idx_neg)           # [n, k_neg]
        view_matched_neg = same_view & neg_mask                # [n, n]
        sel_is_view_matched = view_matched_neg.gather(dim=1, index=idx_neg)  # [n, k_neg]
        sel_is_negative = neg_mask.gather(dim=1, index=idx_neg)              # [n, k_neg]

        w_neg = torch.ones_like(prio_sel)
        if not self.vm_global_topk:
            if self.vm_strict_view_weighting:
                # Anchors with no view-matched negative still learn, but the
                # global hard negative that replaces it only counts aux weight.
                w_neg[~sel_is_view_matched] = self.vm_aux_weight
            else:
                # priority-based aux weighting only applies when priority is used
                w_neg[prio_sel > self.vm_primary_max_priority] = self.vm_aux_weight
        # Anchors owning fewer than k_neg negatives padded the top-k with
        # masked-out entries; drop them instead of letting them drive the loss.
        w_neg = w_neg * sel_is_negative.float()
        w = w_neg.unsqueeze(0).unsqueeze(2)                    # [1, n, 1, k_neg]

        # -- weighted triplet loss --
        dist_diff = ap_dist - an_dist                          # [p, n, K_pos, k_neg]
        raw = F.relu(dist_diff + self.margin)

        raw_flat = raw.view(p, -1)
        w_flat = w.expand_as(raw).contiguous().view(p, -1)
        nz = (raw_flat != 0).float()

        loss_sum = (raw_flat * w_flat).sum(dim=-1)             # [p]
        w_sum = (nz * w_flat).sum(dim=-1)                      # [p]
        loss_avg = loss_sum / (w_sum + 1e-9)
        loss_avg[w_sum == 0] = 0

        hard_loss = raw_flat.max(dim=-1).values                # [p]
        loss_num = nz.sum(dim=-1)                              # [p]
        total_triplets = torch.tensor(float(raw_flat.shape[-1]), device=dist.device).expand(p)  # [p]

        # -- view-match fallback accounting (cheap, logged every step) --
        with torch.no_grad():
            n_sel = max(sel_is_negative.sum().item(), 1)
            self.info['vm/view_matched_neg_ratio'] = \
                sel_is_view_matched.sum() / n_sel
            self.info['vm/global_fallback_neg_ratio'] = \
                (sel_is_negative & ~sel_is_view_matched).sum() / n_sel
            self.info['vm/aux_weighted_neg_ratio'] = \
                ((w_neg < 1.0) & sel_is_negative).sum() / n_sel
            self.info['vm/anchors_without_view_matched_neg_ratio'] = \
                (~view_matched_neg.any(dim=1)).float().mean()
            self.info['vm/valid_view_anchor_ratio'] = valid_view.float().mean()
            self.info['vm/invalid_view_ratio'] = (~valid_view).float().mean()

        # -- logging (every 100 steps) --
        self._vm_log_counter += 1
        if self._vm_log_counter % 100 == 1:
            # legacy logging
            self._vm_log(prio_sel, an_dist, dist.device)
            # new diagnostic logging
            self._vm_log_metrics(
                dist=dist, dist_mean=dist_mean,
                labels=labels, view_id=view_id, outfit_id=outfit_id,
                matches=matches, neg_mask=neg_mask, same_view=same_view,
                ap_dist=ap_dist, an_dist=an_dist, raw=raw,
                idx_neg=idx_neg, k_neg=k_neg)

        return loss_avg, hard_loss, loss_num, total_triplets

    def _view_unmatched_triplet(self, dist, labels, view_id, outfit_id=None):
        """Cross-view (un-matched view) negative mining.

        Negatives are restricted to samples with a DIFFERENT view than the
        anchor (``view_id`` differs). Same-view negatives are NEVER used, which
        is the opposite of the same-view-preferring ``view_match_negative_mining`` mining.

        Two selection policies:
          - top-k (default, ``vm_use_all_unmatched=False``): for each anchor,
            pick the ``vm_k_neg`` hardest (closest) cross-view negatives
            (e.g. k=1 selects the single hardest different-view negative).
          - use-all (``vm_use_all_unmatched=True``): use ALL cross-view
            negatives, ignoring ``vm_k_neg``. Implemented by weighting every
            negative pair (1 for cross-view, 0 for same-view) so the loss is
            driven purely by different-view negatives.
        """
        p, n, _ = dist.size()

        # -- positives: identical to original --
        matches = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()  # [n, n]
        neg_mask = ~matches                                            # [n, n]
        ap_dist = dist[:, matches].view(p, n, -1, 1)                   # [p, n, K_pos, 1]

        cross_view = (view_id.unsqueeze(1) != view_id.unsqueeze(0))    # [n, n]
        cand_mask = neg_mask & cross_view                             # cross-view negatives
        dist_mean = dist.mean(dim=0)                                   # [n, n]

        if self.vm_use_all_unmatched:
            # Use ALL negatives rectangularly (equal count per anchor under PK
            # sampling) and weight same-view negatives to zero.
            an_dist = dist[:, neg_mask].view(p, n, 1, -1)              # [p, n, 1, K_neg]
            w_neg = cross_view[neg_mask].view(n, -1).float()          # [n, K_neg] (1=cross-view)
            idx_neg = None
        else:
            # Top-k hardest cross-view negatives (distance-only).
            min_cross = int(cand_mask.sum(dim=1).min().item())
            k_neg = min(self.vm_k_neg, min_cross)
            if k_neg == 0:
                zero = dist.new_zeros(p)
                return zero, zero, zero, zero
            score = dist_mean.masked_fill(~cand_mask, float("inf"))
            _, idx_neg = torch.topk(score, k=k_neg, dim=1, largest=False)  # [n, k_neg]
            idx_neg_exp = idx_neg.unsqueeze(0).expand(p, -1, -1)           # [p, n, k_neg]
            an_dist = dist.gather(dim=2, index=idx_neg_exp).unsqueeze(2)   # [p, n, 1, k_neg]
            w_neg = torch.ones(n, k_neg, device=dist.device)              # [n, k_neg]

        # -- weighted triplet loss (same machinery as view_match_negative_mining) --
        w = w_neg.unsqueeze(0).unsqueeze(2)                            # [1, n, 1, K]
        dist_diff = ap_dist - an_dist                                  # [p, n, K_pos, K]
        raw = F.relu(dist_diff + self.margin)

        raw_flat = raw.view(p, -1)
        w_flat = w.expand_as(raw).contiguous().view(p, -1)
        nz = (raw_flat != 0).float()

        loss_sum = (raw_flat * w_flat).sum(dim=-1)                     # [p]
        w_sum = (nz * w_flat).sum(dim=-1)                              # [p]
        loss_avg = loss_sum / (w_sum + 1e-9)
        loss_avg[w_sum == 0] = 0

        hard_loss = (raw_flat * w_flat).max(dim=-1).values            # [p] (cross-view only)
        loss_num = w_sum                                              # active cross-view triplets
        total_triplets = w_flat.sum(dim=-1)                          # candidate cross-view triplets

        # -- lightweight logging (every 100 steps) --
        self._vm_log_counter += 1
        if self._vm_log_counter % 100 == 1:
            with torch.no_grad():
                device = dist.device
                if idx_neg is not None:
                    sel_dist = dist_mean.detach().gather(1, idx_neg)
                    self.info['vm/selected_neg_dist_mean'] = sel_dist.mean()
                self.info['vm/selected_neg_cross_view_ratio'] = torch.tensor(
                    1.0, device=device)
                cand_cross = dist_mean.detach()[cand_mask]
                if cand_cross.numel() > 0:
                    self.info['vm/candidate_cross_view_neg_dist_mean'] = \
                        cand_cross.mean()
                self.info['vm/use_all_unmatched'] = torch.tensor(
                    float(self.vm_use_all_unmatched), device=device)

        return loss_avg, hard_loss, loss_num, total_triplets

    def _vm_log(self, prio_sel, an_dist, device):
        """Record legacy view match negative mining statistics.

        Step gating is handled by the caller (every 100 steps).
        """
        total = max(prio_sel.numel(), 1)
        for p_val in range(4):
            ratio = (prio_sel == p_val).float().sum().item() / total
            self.info[f'vm_prio{p_val}_ratio'] = torch.tensor(
                ratio, device=device)

        primary_mask = (prio_sel <= self.vm_primary_max_priority)
        aux_mask = ~primary_mask
        an_mean = an_dist.mean(dim=0).squeeze(1)  # [n, k_neg]
        if primary_mask.any():
            self.info['vm_primary_neg_dist'] = torch.tensor(
                an_mean[primary_mask].mean().item(), device=device)
        if aux_mask.any():
            self.info['vm_aux_neg_dist'] = torch.tensor(
                an_mean[aux_mask].mean().item(), device=device)

    def _vm_log_metrics(self, dist, dist_mean, labels, view_id, outfit_id,
                        matches, neg_mask, same_view,
                        ap_dist, an_dist, raw, idx_neg, k_neg):
        """Compute diagnostic metrics for analyzing view match negative mining vs global
        top-k mining.

        All metrics are computed under ``torch.no_grad`` and stored as scalar
        tensors on ``self.info`` so they flow through ``LossAggregator`` and
        appear as ``scalar/triplet/<key>`` in the trainer log / tensorboard.

        The implementation reuses the already-computed pairwise distance
        matrix and avoids any Python-level loops over the batch.
        """
        with torch.no_grad():
            info = self.info
            device = dist.device
            p, n, _ = dist.size()

            dist_mean_d = dist_mean.detach()
            same_view_d = same_view.detach()
            diff_view_d = ~same_view_d

            # Self-pair (same anchor) is excluded from positive stats.
            eye = torch.eye(n, dtype=torch.bool, device=device)
            pos_pair = matches & ~eye  # [n, n]

            def _safe_mean(t):
                return t.float().mean() if t.numel() > 0 else \
                    torch.tensor(float('nan'), device=device)

            def _safe_std(t):
                return t.float().std() if t.numel() > 1 else \
                    torch.tensor(float('nan'), device=device)

            # -- [3] Positive distance statistics --
            pos_d = dist_mean_d[pos_pair]
            info['pos/dist_mean'] = _safe_mean(pos_d)
            info['pos/dist_std'] = _safe_std(pos_d)

            same_v_pos = pos_pair & same_view_d
            cross_v_pos = pos_pair & diff_view_d
            if same_v_pos.any():
                info['pos/same_view_dist_mean'] = dist_mean_d[same_v_pos].mean()
            if cross_v_pos.any():
                info['pos/cross_view_dist_mean'] = dist_mean_d[cross_v_pos].mean()

            # outfit-conditioned positive stats (only if outfit metadata
            # is available)
            same_outfit_full = None
            if outfit_id is not None:
                same_outfit_full = (
                    outfit_id.unsqueeze(1) == outfit_id.unsqueeze(0))
                same_o_pos = pos_pair & same_outfit_full
                cross_o_pos = pos_pair & ~same_outfit_full
                if same_o_pos.any():
                    info['pos/same_outfit_dist_mean'] = \
                        dist_mean_d[same_o_pos].mean()
                if cross_o_pos.any():
                    info['pos/cross_outfit_dist_mean'] = \
                        dist_mean_d[cross_o_pos].mean()

            # -- [4] Negative distance statistics --
            neg_d = dist_mean_d[neg_mask]
            info['neg/dist_mean'] = _safe_mean(neg_d)
            info['neg/dist_std'] = _safe_std(neg_d)

            same_v_neg = neg_mask & same_view_d
            cross_v_neg = neg_mask & diff_view_d
            if same_v_neg.any():
                info['neg/same_view_dist_mean'] = dist_mean_d[same_v_neg].mean()
            if cross_v_neg.any():
                info['neg/cross_view_dist_mean'] = dist_mean_d[cross_v_neg].mean()

            # -- [2] Candidate negative statistics --
            n_neg_total = neg_mask.sum().clamp(min=1).float()
            info['vm/candidate_neg_same_view_ratio'] = \
                same_v_neg.sum().float() / n_neg_total
            info['vm/candidate_neg_cross_view_ratio'] = \
                cross_v_neg.sum().float() / n_neg_total
            if same_v_neg.any():
                info['vm/candidate_same_view_neg_dist_mean'] = \
                    dist_mean_d[same_v_neg].mean()
            if cross_v_neg.any():
                info['vm/candidate_cross_view_neg_dist_mean'] = \
                    dist_mean_d[cross_v_neg].mean()

            # -- [1] Selected negative statistics --
            anchor_view = view_id.unsqueeze(1).expand_as(idx_neg)  # [n, k_neg]
            sel_view = view_id[idx_neg]                            # [n, k_neg]
            sel_same_view = (anchor_view == sel_view)
            sel_cross_view = ~sel_same_view

            sel_dist = dist_mean_d.gather(1, idx_neg)              # [n, k_neg]

            info['vm/selected_neg_same_view_ratio'] = \
                sel_same_view.float().mean()
            info['vm/selected_neg_cross_view_ratio'] = \
                sel_cross_view.float().mean()
            info['vm/selected_neg_dist_mean'] = sel_dist.mean()
            info['vm/selected_neg_dist_std'] = _safe_std(sel_dist.flatten())
            # Analysis priority: 0 = same view, 1 = cross view; lower means
            # mining is more same-view-centric. Independent from the internal
            # (use_viewset-aware) priority used to drive top-k.
            analysis_prio = sel_cross_view.float()                 # [n, k_neg]
            info['vm/selected_neg_priority_mean'] = analysis_prio.mean()

            # -- [5] Margin statistics --
            # Per-triplet margin = mean_part(d_neg) - mean_part(d_pos).
            ap_mean = ap_dist.detach().mean(dim=0)  # [n, K_pos, 1]
            an_mean = an_dist.detach().mean(dim=0)  # [n, 1, k_neg]
            margin_per_triplet = an_mean - ap_mean  # [n, K_pos, k_neg]

            info['vm/margin_mean'] = margin_per_triplet.mean()
            info['vm/margin_std'] = _safe_std(margin_per_triplet.flatten())

            sel_same_v_b = sel_same_view.unsqueeze(1).expand_as(margin_per_triplet)
            sel_cross_v_b = ~sel_same_v_b
            if sel_same_v_b.any():
                info['vm/same_view_neg_margin_mean'] = \
                    margin_per_triplet[sel_same_v_b].mean()
            if sel_cross_v_b.any():
                info['vm/cross_view_neg_margin_mean'] = \
                    margin_per_triplet[sel_cross_v_b].mean()

            if same_outfit_full is not None:
                # Reconstruct outfit equality for the K_pos columns kept in
                # ap_dist. ``matches`` selects the same K_pos positions per
                # anchor when reshaped, so the same indexing works here.
                same_o_per_pos = same_outfit_full[matches].view(n, -1)  # [n, K_pos]
                same_o_b = same_o_per_pos.unsqueeze(2).expand_as(margin_per_triplet)
                cross_o_b = ~same_o_b
                if same_o_b.any():
                    info['vm/same_outfit_pos_margin_mean'] = \
                        margin_per_triplet[same_o_b].mean()
                if cross_o_b.any():
                    info['vm/cross_outfit_pos_margin_mean'] = \
                        margin_per_triplet[cross_o_b].mean()

            # -- [6] Hinge / active triplet statistics --
            hinge_per_triplet = raw.detach().mean(dim=0)  # [n, K_pos, k_neg]
            info['vm/hinge_loss_mean'] = hinge_per_triplet.mean()
            info['vm/hinge_loss_std'] = _safe_std(hinge_per_triplet.flatten())

            active = (hinge_per_triplet > 0)
            n_total_triplets = float(active.numel())
            info['vm/active_triplet_ratio'] = active.float().mean()
            n_active = active.sum().float()
            denom = n_active.clamp(min=1.0)
            info['vm/active_same_view_neg_ratio'] = \
                (active & sel_same_v_b).float().sum() / denom
            info['vm/active_cross_view_neg_ratio'] = \
                (active & sel_cross_v_b).float().sum() / denom

            # -- [7] Global top-k comparison (analysis only) --
            global_score = dist_mean_d.masked_fill(~neg_mask, float('inf'))
            _, idx_global = torch.topk(
                global_score, k=k_neg, dim=1, largest=False)
            global_view = view_id[idx_global]
            global_same_v = (anchor_view == global_view)
            global_cross_v = ~global_same_v
            global_dist = dist_mean_d.gather(1, idx_global)        # [n, k_neg]

            info['cmp/global_topk_same_view_ratio'] = \
                global_same_v.float().mean()
            info['cmp/global_topk_cross_view_ratio'] = \
                global_cross_v.float().mean()
            info['cmp/global_topk_dist_mean'] = global_dist.mean()
            info['cmp/vm_topk_dist_mean'] = sel_dist.mean()
            info['cmp/global_minus_vm_dist_mean'] = \
                global_dist.mean() - sel_dist.mean()

            # Hinge / margin under the alternative (global top-k) negatives.
            idx_global_exp = idx_global.unsqueeze(0).expand(p, -1, -1)
            an_dist_global = dist.detach().gather(
                2, idx_global_exp).unsqueeze(2)                    # [p, n, 1, k_neg]
            an_mean_global = an_dist_global.mean(dim=0)            # [n, 1, k_neg]
            margin_global = an_mean_global - ap_mean               # [n, K_pos, k_neg]
            hinge_global = F.relu(
                ap_dist.detach() - an_dist_global + self.margin
            ).mean(dim=0)                                          # [n, K_pos, k_neg]

            info['cmp/global_topk_margin_mean'] = margin_global.mean()
            info['cmp/vm_topk_margin_mean'] = margin_per_triplet.mean()
            info['cmp/global_topk_hinge_mean'] = hinge_global.mean()
            info['cmp/vm_topk_hinge_mean'] = hinge_per_triplet.mean()
            info['cmp/global_topk_hinge_std'] = \
                _safe_std(hinge_global.flatten())
            info['cmp/vm_topk_hinge_std'] = \
                _safe_std(hinge_per_triplet.flatten())

            # Touch unused locals to keep linters quiet without changing
            # numerical behaviour.
            _ = n_total_triplets

    # --- original helpers (unchanged) ---

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist
