import os
from time import strftime, localtime
import csv
import numpy as np
from utils import get_msg_mgr, mkdir

from .metric import mean_iou, cuda_dist, compute_ACC_mAP, evaluate_rank, evaluate_many
from .re_rank import re_ranking

def de_diag(acc, each_angle=False):
    # Exclude identical-view cases
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result


def _parse_ccpg_type(seq_type):
    parts = str(seq_type).split("_")
    upper = next((p for p in parts if p.startswith("U")), "")
    lower = next((p for p in parts if p.startswith("D")), "")
    bg = "BG" if "BG" in parts else "NO_BG"
    return upper, lower, bg


def _safe_fraction(numer, denom):
    return float(numer) / float(denom) if denom else float("nan")


def _safe_lift(value, chance):
    if chance and not np.isnan(chance) and chance > 0 and not np.isnan(value):
        return value / chance
    return float("nan")


def _mean_or_nan(values):
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def _ccpg_error_covariate_row(setting, n_valid, n_error, same_counts, chance_values):
    row = {
        "setting": setting,
        "n_valid": int(n_valid),
        "n_error": int(n_error),
        "rank1_acc": 1.0 - _safe_fraction(n_error, n_valid) if n_valid else float("nan"),
    }
    for key in ("view", "type", "upper", "lower", "bg"):
        frac = _safe_fraction(same_counts[key], n_error)
        chance = _mean_or_nan(chance_values[key])
        row[f"same_{key}_errors"] = int(same_counts[key])
        row[f"same_{key}_among_errors"] = frac
        row[f"chance_same_{key}"] = chance
        row[f"{key}_bias_lift"] = _safe_lift(frac, chance)
    return row


def _ccpg_error_covariate_overall(rows):
    valid = sum(r["n_valid"] for r in rows)
    errors = sum(r["n_error"] for r in rows)
    row = {
        "setting": "OVERALL",
        "n_valid": int(valid),
        "n_error": int(errors),
        "rank1_acc": 1.0 - _safe_fraction(errors, valid) if valid else float("nan"),
    }
    for key in ("view", "type", "upper", "lower", "bg"):
        count_key = f"same_{key}_errors"
        chance_key = f"chance_same_{key}"
        frac = _safe_fraction(sum(r[count_key] for r in rows), errors)
        weighted = []
        weights = []
        for r in rows:
            if not np.isnan(r[chance_key]):
                weighted.append(r[chance_key])
                weights.append(r["n_valid"])
        chance = float(np.average(weighted, weights=weights)) if weights else float("nan")
        row[count_key] = int(sum(r[count_key] for r in rows))
        row[f"same_{key}_among_errors"] = frac
        row[chance_key] = chance
        row[f"{key}_bias_lift"] = _safe_lift(frac, chance)
    return row


def _ccpg_rank1_error_covariates(distmat, probe_y, gallery_y,
                                 probe_view, gallery_view,
                                 probe_type, gallery_type, setting):
    order = np.argsort(distmat, axis=1)
    probe_upper = np.asarray([_parse_ccpg_type(t)[0] for t in probe_type])
    probe_lower = np.asarray([_parse_ccpg_type(t)[1] for t in probe_type])
    probe_bg = np.asarray([_parse_ccpg_type(t)[2] for t in probe_type])
    gallery_upper = np.asarray([_parse_ccpg_type(t)[0] for t in gallery_type])
    gallery_lower = np.asarray([_parse_ccpg_type(t)[1] for t in gallery_type])
    gallery_bg = np.asarray([_parse_ccpg_type(t)[2] for t in gallery_type])

    same_counts = {k: 0 for k in ("view", "type", "upper", "lower", "bg")}
    chance_values = {k: [] for k in ("view", "type", "upper", "lower", "bg")}
    cases = []
    n_valid = 0
    n_error = 0

    for q_idx in range(distmat.shape[0]):
        sorted_idx = order[q_idx]
        remove = (gallery_y[sorted_idx] == probe_y[q_idx]) & (
            gallery_view[sorted_idx] == probe_view[q_idx])
        kept = sorted_idx[~remove]
        if kept.size == 0:
            continue
        if not np.any(gallery_y[kept] == probe_y[q_idx]):
            continue
        n_valid += 1

        diff_id = gallery_y[kept] != probe_y[q_idx]
        if np.any(diff_id):
            diff_kept = kept[diff_id]
            chance_values["view"].append(float(np.mean(gallery_view[diff_kept] == probe_view[q_idx])))
            chance_values["type"].append(float(np.mean(gallery_type[diff_kept] == probe_type[q_idx])))
            chance_values["upper"].append(float(np.mean(gallery_upper[diff_kept] == probe_upper[q_idx])))
            chance_values["lower"].append(float(np.mean(gallery_lower[diff_kept] == probe_lower[q_idx])))
            chance_values["bg"].append(float(np.mean(gallery_bg[diff_kept] == probe_bg[q_idx])))

        top = kept[0]
        if gallery_y[top] == probe_y[q_idx]:
            continue

        n_error += 1
        flags = {
            "view": gallery_view[top] == probe_view[q_idx],
            "type": gallery_type[top] == probe_type[q_idx],
            "upper": gallery_upper[top] == probe_upper[q_idx],
            "lower": gallery_lower[top] == probe_lower[q_idx],
            "bg": gallery_bg[top] == probe_bg[q_idx],
        }
        for key, value in flags.items():
            same_counts[key] += int(value)
        cases.append({
            "protocol": "official_ccpg",
            "setting": setting,
            "query_local_index": int(q_idx),
            "top1_gallery_local_index": int(top),
            "query_pid": str(probe_y[q_idx]),
            "top1_pid": str(gallery_y[top]),
            "query_type": str(probe_type[q_idx]),
            "top1_type": str(gallery_type[top]),
            "query_view": str(probe_view[q_idx]),
            "top1_view": str(gallery_view[top]),
            "same_view": int(flags["view"]),
            "same_type": int(flags["type"]),
            "same_upper": int(flags["upper"]),
            "same_lower": int(flags["lower"]),
            "same_bg": int(flags["bg"]),
        })

    row = _ccpg_error_covariate_row(
        setting, n_valid, n_error, same_counts, chance_values)
    return row, cases


def _ccpg_top1_nn_indices(feature, metric="euc", chunk=256):
    import torch
    import torch.nn.functional as F

    feat = torch.from_numpy(np.asarray(feature)).cuda()
    if metric == "cos":
        feat = F.normalize(feat, p=2, dim=1)
    num_bin = feat.size(2) if feat.dim() == 3 else 1
    n = feat.size(0)
    nn_idx = []
    with torch.no_grad():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            q = feat[start:end]
            dist = torch.zeros(end - start, n).cuda()
            for b in range(num_bin):
                q_b = q[:, :, b] if feat.dim() == 3 else q
                f_b = feat[:, :, b] if feat.dim() == 3 else feat
                if metric == "cos":
                    dist += torch.matmul(q_b, f_b.transpose(0, 1))
                else:
                    d = torch.sum(q_b ** 2, 1).unsqueeze(1) + torch.sum(f_b ** 2, 1).unsqueeze(
                        0) - 2 * torch.matmul(q_b, f_b.transpose(0, 1))
                    dist += torch.sqrt(torch.relu(d))
            dist = 1 - dist / num_bin if metric == "cos" else dist / num_bin
            rows = torch.arange(end - start, device=dist.device)
            cols = torch.arange(start, end, device=dist.device)
            dist[rows, cols] = float("inf")
            nn_idx.append(dist.argmin(dim=1).cpu().numpy())
    return np.concatenate(nn_idx)


def _ccpg_global_nn_error_covariates(feature, label, seq_type, view_np, metric="euc"):
    label = np.asarray(label)
    seq_type = np.asarray(seq_type)
    view_np = np.asarray(view_np)
    upper = np.asarray([_parse_ccpg_type(t)[0] for t in seq_type])
    lower = np.asarray([_parse_ccpg_type(t)[1] for t in seq_type])
    bg = np.asarray([_parse_ccpg_type(t)[2] for t in seq_type])
    nn_idx = _ccpg_top1_nn_indices(feature, metric)
    wrong = label[nn_idx] != label

    same_counts = {
        "view": int(np.sum((view_np[nn_idx] == view_np) & wrong)),
        "type": int(np.sum((seq_type[nn_idx] == seq_type) & wrong)),
        "upper": int(np.sum((upper[nn_idx] == upper) & wrong)),
        "lower": int(np.sum((lower[nn_idx] == lower) & wrong)),
        "bg": int(np.sum((bg[nn_idx] == bg) & wrong)),
    }
    chance_values = {k: [] for k in ("view", "type", "upper", "lower", "bg")}
    for i in range(len(label)):
        diff_id = label != label[i]
        if not np.any(diff_id):
            continue
        chance_values["view"].append(float(np.mean(view_np[diff_id] == view_np[i])))
        chance_values["type"].append(float(np.mean(seq_type[diff_id] == seq_type[i])))
        chance_values["upper"].append(float(np.mean(upper[diff_id] == upper[i])))
        chance_values["lower"].append(float(np.mean(lower[diff_id] == lower[i])))
        chance_values["bg"].append(float(np.mean(bg[diff_id] == bg[i])))

    cases = []
    for q_idx in np.flatnonzero(wrong):
        top = nn_idx[q_idx]
        cases.append({
            "protocol": "global_nn",
            "setting": "OVERALL",
            "query_local_index": int(q_idx),
            "top1_gallery_local_index": int(top),
            "query_pid": str(label[q_idx]),
            "top1_pid": str(label[top]),
            "query_type": str(seq_type[q_idx]),
            "top1_type": str(seq_type[top]),
            "query_view": str(view_np[q_idx]),
            "top1_view": str(view_np[top]),
            "same_view": int(view_np[top] == view_np[q_idx]),
            "same_type": int(seq_type[top] == seq_type[q_idx]),
            "same_upper": int(upper[top] == upper[q_idx]),
            "same_lower": int(lower[top] == lower[q_idx]),
            "same_bg": int(bg[top] == bg[q_idx]),
        })

    row = _ccpg_error_covariate_row(
        "OVERALL", len(label), int(np.sum(wrong)), same_counts, chance_values)
    return [row], cases


_CCPG_POSITIVE_RANK_TOPKS = (10, 20, 50, 100)


def _nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _nanpercentile(values, q):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


def _ccpg_positive_rank_summary(setting, query_rows):
    row = {"setting": setting, "n_valid": int(len(query_rows))}
    if not query_rows:
        fields = [
            "rank1_acc", "mAP", "mINP", "positives_per_query_mean",
            "candidate_gallery_size_mean", "first_positive_rank_mean",
            "first_positive_rank_median", "first_positive_rank_p90",
            "mean_positive_rank_mean", "mean_impostors_before_positive_mean",
            "last_positive_rank_mean", "last_positive_rank_median",
            "last_positive_rank_p90", "last_positive_percentile_mean",
            "last_positive_percentile_p90", "tail_impostors_before_last_positive_mean",
        ]
        for key in fields:
            row[key] = float("nan")
        for k in _CCPG_POSITIVE_RANK_TOPKS:
            row[f"positive_coverage_at_{k}"] = float("nan")
            row[f"all_positives_in_top{k}_frac"] = float("nan")
        return row

    def vals(key):
        return [r[key] for r in query_rows]

    row.update({
        "rank1_acc": _nanmean(vals("rank1")),
        "mAP": _nanmean(vals("ap")),
        "mINP": _nanmean(vals("inp")),
        "positives_per_query_mean": _nanmean(vals("num_rel")),
        "candidate_gallery_size_mean": _nanmean(vals("n_gallery")),
        "first_positive_rank_mean": _nanmean(vals("first_positive_rank")),
        "first_positive_rank_median": _nanpercentile(vals("first_positive_rank"), 50),
        "first_positive_rank_p90": _nanpercentile(vals("first_positive_rank"), 90),
        "mean_positive_rank_mean": _nanmean(vals("mean_positive_rank")),
        "mean_impostors_before_positive_mean": _nanmean(
            vals("mean_impostors_before_positive")),
        "last_positive_rank_mean": _nanmean(vals("last_positive_rank")),
        "last_positive_rank_median": _nanpercentile(vals("last_positive_rank"), 50),
        "last_positive_rank_p90": _nanpercentile(vals("last_positive_rank"), 90),
        "last_positive_percentile_mean": _nanmean(vals("last_positive_percentile")),
        "last_positive_percentile_p90": _nanpercentile(vals("last_positive_percentile"), 90),
        "tail_impostors_before_last_positive_mean": _nanmean(
            vals("tail_impostors_before_last_positive")),
    })
    for k in _CCPG_POSITIVE_RANK_TOPKS:
        row[f"positive_coverage_at_{k}"] = _nanmean(vals(f"coverage_at_{k}"))
        row[f"all_positives_in_top{k}_frac"] = _nanmean(vals(f"all_top{k}"))
    return row


def _ccpg_positive_rank_diagnostics(distmat, probe_y, gallery_y,
                                    probe_view, gallery_view, setting,
                                    probe_type=None):
    """How completely the correct identity is pulled forward in the rank list."""
    order = np.argsort(distmat, axis=1)
    query_rows = []

    for q_idx in range(distmat.shape[0]):
        sorted_idx = order[q_idx]
        remove = (gallery_y[sorted_idx] == probe_y[q_idx]) & (
            gallery_view[sorted_idx] == probe_view[q_idx])
        kept = sorted_idx[~remove]
        if kept.size == 0:
            continue
        matches = (gallery_y[kept] == probe_y[q_idx]).astype(np.int32)
        if not np.any(matches):
            continue

        pos_idx = np.flatnonzero(matches)
        num_rel = int(pos_idx.size)
        first_rank = int(pos_idx[0] + 1)
        last_rank = int(pos_idx[-1] + 1)
        positive_ranks = pos_idx + 1
        impostors_before_positive = positive_ranks - np.arange(1, num_rel + 1)
        cumulative = matches.cumsum()
        precision_at_pos = cumulative[pos_idx] / (pos_idx + 1.0)

        q_row = {
            "setting": setting,
            "query_local_index": int(q_idx),
            "query_pid": str(probe_y[q_idx]),
            "query_type": "" if probe_type is None else str(probe_type[q_idx]),
            "query_view": str(probe_view[q_idx]),
            "rank1": float(matches[0]),
            "ap": float(np.mean(precision_at_pos)),
            "inp": float(num_rel / last_rank),
            "num_rel": num_rel,
            "n_gallery": int(kept.size),
            "first_positive_rank": first_rank,
            "mean_positive_rank": float(np.mean(positive_ranks)),
            "mean_impostors_before_positive": float(
                np.mean(impostors_before_positive)),
            "last_positive_rank": last_rank,
            "last_positive_percentile": float(last_rank / kept.size),
            "tail_impostors_before_last_positive": float(last_rank - num_rel),
        }
        for k in _CCPG_POSITIVE_RANK_TOPKS:
            q_row[f"coverage_at_{k}"] = float(matches[:k].sum() / num_rel)
            q_row[f"all_top{k}"] = float(last_rank <= k)
        query_rows.append(q_row)

    return _ccpg_positive_rank_summary(setting, query_rows), query_rows


def _ccpg_rank1_cases(distmat, probe_y, gallery_y,
                      probe_view, gallery_view,
                      probe_type, gallery_type,
                      probe_indices, gallery_indices, setting):
    """Per-valid-query rank-1 retrieval rows under the official CCPG filter."""
    order = np.argsort(distmat, axis=1)
    rows = []

    for q_idx in range(distmat.shape[0]):
        sorted_idx = order[q_idx]
        remove = (gallery_y[sorted_idx] == probe_y[q_idx]) & (
            gallery_view[sorted_idx] == probe_view[q_idx])
        kept = sorted_idx[~remove]
        if kept.size == 0:
            continue
        matches = gallery_y[kept] == probe_y[q_idx]
        if not np.any(matches):
            continue

        top = int(kept[0])
        pos_offsets = np.flatnonzero(matches)
        first_pos = int(kept[int(pos_offsets[0])])
        rows.append({
            "setting": setting,
            "query_local_index": int(q_idx),
            "query_global_index": int(probe_indices[q_idx]),
            "query_pid": str(probe_y[q_idx]),
            "query_type": str(probe_type[q_idx]),
            "query_view": str(probe_view[q_idx]),
            "top1_gallery_local_index": top,
            "top1_gallery_global_index": int(gallery_indices[top]),
            "top1_pid": str(gallery_y[top]),
            "top1_type": str(gallery_type[top]),
            "top1_view": str(gallery_view[top]),
            "top1_dist": float(distmat[q_idx, top]),
            "rank1": int(gallery_y[top] == probe_y[q_idx]),
            "first_positive_rank": int(pos_offsets[0] + 1),
            "first_positive_gallery_local_index": first_pos,
            "first_positive_gallery_global_index": int(gallery_indices[first_pos]),
            "first_positive_type": str(gallery_type[first_pos]),
            "first_positive_view": str(gallery_view[first_pos]),
            "first_positive_dist": float(distmat[q_idx, first_pos]),
            "num_rel": int(np.sum(matches)),
            "n_gallery": int(kept.size),
        })
    return rows


def _write_ccpg_positive_rank_diagnostics(save_dir, rows, query_rows=None):
    mkdir(save_dir)
    fieldnames = [
        "setting", "n_valid", "rank1_acc", "mAP", "mINP",
        "positives_per_query_mean", "candidate_gallery_size_mean",
        "first_positive_rank_mean", "first_positive_rank_median",
        "first_positive_rank_p90", "mean_positive_rank_mean",
        "mean_impostors_before_positive_mean", "last_positive_rank_mean",
        "last_positive_rank_median", "last_positive_rank_p90",
        "last_positive_percentile_mean", "last_positive_percentile_p90",
        "tail_impostors_before_last_positive_mean",
    ]
    for k in _CCPG_POSITIVE_RANK_TOPKS:
        fieldnames.extend([f"positive_coverage_at_{k}",
                           f"all_positives_in_top{k}_frac"])
    with open(os.path.join(save_dir, "ccpg_positive_rank_diagnostics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if query_rows is None:
        return
    query_fieldnames = [
        "setting", "query_local_index", "query_pid", "query_type",
        "query_view", "rank1", "ap", "inp", "num_rel", "n_gallery",
        "first_positive_rank", "mean_positive_rank",
        "mean_impostors_before_positive", "last_positive_rank",
        "last_positive_percentile", "tail_impostors_before_last_positive",
    ]
    for k in _CCPG_POSITIVE_RANK_TOPKS:
        query_fieldnames.extend([f"coverage_at_{k}", f"all_top{k}"])
    with open(os.path.join(save_dir, "ccpg_positive_rank_query_diagnostics.csv"),
              "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=query_fieldnames)
        writer.writeheader()
        writer.writerows(query_rows)


def _write_ccpg_rank1_cases(save_dir, rows):
    mkdir(save_dir)
    fieldnames = [
        "setting", "query_local_index", "query_global_index",
        "query_pid", "query_type", "query_view",
        "top1_gallery_local_index", "top1_gallery_global_index",
        "top1_pid", "top1_type", "top1_view", "top1_dist", "rank1",
        "first_positive_rank", "first_positive_gallery_local_index",
        "first_positive_gallery_global_index", "first_positive_type",
        "first_positive_view", "first_positive_dist", "num_rel", "n_gallery",
    ]
    with open(os.path.join(save_dir, "ccpg_rank1_cases.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ccpg_identity_margin_summary(setting, query_rows):
    row = {"setting": setting, "n_valid": int(len(query_rows))}
    if not query_rows:
        for key in [
            "violation_count", "violation_rate", "margin_mean",
            "margin_median", "margin_p10", "violation_gap_mean",
            "impostors_before_first_positive_mean",
            "impostors_before_first_positive_p90",
            "hard_neg_same_view_frac", "hard_neg_same_view_frac_violations",
            "hard_neg_cross_view_frac_violations",
            "hard_neg_same_upper_frac_violations",
            "hard_neg_same_lower_frac_violations",
            "hard_neg_same_bg_frac_violations",
            "hard_pos_same_upper_frac", "hard_pos_same_lower_frac",
            "hard_pos_same_bg_frac",
        ]:
            row[key] = float("nan")
        return row

    def vals(key):
        return [r[key] for r in query_rows]

    violations = [r for r in query_rows if r["violation"]]
    row.update({
        "violation_count": int(len(violations)),
        "violation_rate": _safe_fraction(len(violations), len(query_rows)),
        "margin_mean": _nanmean(vals("identity_margin")),
        "margin_median": _nanpercentile(vals("identity_margin"), 50),
        "margin_p10": _nanpercentile(vals("identity_margin"), 10),
        "violation_gap_mean": _nanmean(vals("violation_gap")),
        "impostors_before_first_positive_mean": _nanmean(
            vals("impostors_before_first_positive")),
        "impostors_before_first_positive_p90": _nanpercentile(
            vals("impostors_before_first_positive"), 90),
        "hard_neg_same_view_frac": _nanmean(vals("hard_neg_same_view")),
        "hard_pos_same_upper_frac": _nanmean(vals("hard_pos_same_upper")),
        "hard_pos_same_lower_frac": _nanmean(vals("hard_pos_same_lower")),
        "hard_pos_same_bg_frac": _nanmean(vals("hard_pos_same_bg")),
    })

    if violations:
        def vvals(key):
            return [r[key] for r in violations]
        row.update({
            "hard_neg_same_view_frac_violations": _nanmean(
                vvals("hard_neg_same_view")),
            "hard_neg_cross_view_frac_violations": 1.0 - _nanmean(
                vvals("hard_neg_same_view")),
            "hard_neg_same_upper_frac_violations": _nanmean(
                vvals("hard_neg_same_upper")),
            "hard_neg_same_lower_frac_violations": _nanmean(
                vvals("hard_neg_same_lower")),
            "hard_neg_same_bg_frac_violations": _nanmean(
                vvals("hard_neg_same_bg")),
        })
    else:
        row.update({
            "hard_neg_same_view_frac_violations": float("nan"),
            "hard_neg_cross_view_frac_violations": float("nan"),
            "hard_neg_same_upper_frac_violations": float("nan"),
            "hard_neg_same_lower_frac_violations": float("nan"),
            "hard_neg_same_bg_frac_violations": float("nan"),
        })
    return row


def _ccpg_identity_margin_diagnostics(distmat, probe_y, gallery_y,
                                      probe_view, gallery_view,
                                      probe_type, gallery_type, setting):
    """Closest-positive vs closest-negative margins after official filtering."""
    query_rows = []
    case_rows = []
    probe_upper = np.asarray([_parse_ccpg_type(t)[0] for t in probe_type])
    probe_lower = np.asarray([_parse_ccpg_type(t)[1] for t in probe_type])
    probe_bg = np.asarray([_parse_ccpg_type(t)[2] for t in probe_type])
    gallery_upper = np.asarray([_parse_ccpg_type(t)[0] for t in gallery_type])
    gallery_lower = np.asarray([_parse_ccpg_type(t)[1] for t in gallery_type])
    gallery_bg = np.asarray([_parse_ccpg_type(t)[2] for t in gallery_type])

    for q_idx in range(distmat.shape[0]):
        remove = (gallery_y == probe_y[q_idx]) & (
            gallery_view == probe_view[q_idx])
        kept = np.flatnonzero(~remove)
        if kept.size == 0:
            continue

        matches = gallery_y[kept] == probe_y[q_idx]
        if not np.any(matches) or np.all(matches):
            continue

        pos_idx = kept[matches]
        neg_idx = kept[~matches]
        pos_d = distmat[q_idx, pos_idx]
        neg_d = distmat[q_idx, neg_idx]
        hard_pos = pos_idx[int(np.argmin(pos_d))]
        hard_neg = neg_idx[int(np.argmin(neg_d))]
        hard_pos_dist = float(distmat[q_idx, hard_pos])
        hard_neg_dist = float(distmat[q_idx, hard_neg])
        margin = hard_neg_dist - hard_pos_dist
        violation = margin <= 0.0
        impostors = int(np.sum(neg_d < hard_pos_dist))

        row = {
            "setting": setting,
            "query_local_index": int(q_idx),
            "query_pid": str(probe_y[q_idx]),
            "query_type": str(probe_type[q_idx]),
            "query_view": str(probe_view[q_idx]),
            "hard_pos_gallery_local_index": int(hard_pos),
            "hard_pos_type": str(gallery_type[hard_pos]),
            "hard_pos_view": str(gallery_view[hard_pos]),
            "hard_pos_dist": hard_pos_dist,
            "hard_neg_gallery_local_index": int(hard_neg),
            "hard_neg_pid": str(gallery_y[hard_neg]),
            "hard_neg_type": str(gallery_type[hard_neg]),
            "hard_neg_view": str(gallery_view[hard_neg]),
            "hard_neg_dist": hard_neg_dist,
            "identity_margin": float(margin),
            "violation": int(violation),
            "violation_gap": float(max(-margin, 0.0)),
            "impostors_before_first_positive": impostors,
            "hard_neg_same_view": int(gallery_view[hard_neg] == probe_view[q_idx]),
            "hard_neg_same_upper": int(gallery_upper[hard_neg] == probe_upper[q_idx]),
            "hard_neg_same_lower": int(gallery_lower[hard_neg] == probe_lower[q_idx]),
            "hard_neg_same_bg": int(gallery_bg[hard_neg] == probe_bg[q_idx]),
            "hard_pos_same_upper": int(gallery_upper[hard_pos] == probe_upper[q_idx]),
            "hard_pos_same_lower": int(gallery_lower[hard_pos] == probe_lower[q_idx]),
            "hard_pos_same_bg": int(gallery_bg[hard_pos] == probe_bg[q_idx]),
        }
        query_rows.append(row)
        if violation:
            case_rows.append(row)

    return _ccpg_identity_margin_summary(setting, query_rows), query_rows, case_rows


def _write_ccpg_identity_margin_diagnostics(save_dir, rows, cases):
    mkdir(save_dir)
    fieldnames = [
        "setting", "n_valid", "violation_count", "violation_rate",
        "margin_mean", "margin_median", "margin_p10",
        "violation_gap_mean",
        "impostors_before_first_positive_mean",
        "impostors_before_first_positive_p90",
        "hard_neg_same_view_frac",
        "hard_neg_same_view_frac_violations",
        "hard_neg_cross_view_frac_violations",
        "hard_neg_same_upper_frac_violations",
        "hard_neg_same_lower_frac_violations",
        "hard_neg_same_bg_frac_violations",
        "hard_pos_same_upper_frac", "hard_pos_same_lower_frac",
        "hard_pos_same_bg_frac",
    ]
    with open(os.path.join(save_dir, "ccpg_identity_margin_diagnostics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    case_fieldnames = [
        "setting", "query_local_index", "query_pid", "query_type",
        "query_view", "hard_pos_gallery_local_index", "hard_pos_type",
        "hard_pos_view", "hard_pos_dist", "hard_neg_gallery_local_index",
        "hard_neg_pid", "hard_neg_type", "hard_neg_view", "hard_neg_dist",
        "identity_margin", "violation", "violation_gap",
        "impostors_before_first_positive", "hard_neg_same_view",
        "hard_neg_same_upper", "hard_neg_same_lower", "hard_neg_same_bg",
        "hard_pos_same_upper", "hard_pos_same_lower", "hard_pos_same_bg",
    ]
    with open(os.path.join(save_dir, "ccpg_identity_margin_violation_cases.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=case_fieldnames)
        writer.writeheader()
        writer.writerows(cases)


def _write_ccpg_error_covariates(save_dir, official_rows, official_cases,
                                 global_rows=None, global_cases=None):
    mkdir(save_dir)
    fieldnames = [
        "setting", "n_valid", "n_error", "rank1_acc",
        "same_view_errors", "same_view_among_errors", "chance_same_view", "view_bias_lift",
        "same_type_errors", "same_type_among_errors", "chance_same_type", "type_bias_lift",
        "same_upper_errors", "same_upper_among_errors", "chance_same_upper", "upper_bias_lift",
        "same_lower_errors", "same_lower_among_errors", "chance_same_lower", "lower_bias_lift",
        "same_bg_errors", "same_bg_among_errors", "chance_same_bg", "bg_bias_lift",
    ]
    with open(os.path.join(save_dir, "ccpg_error_covariates.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(official_rows)

    case_fieldnames = [
        "protocol", "setting", "query_local_index", "top1_gallery_local_index",
        "query_pid", "top1_pid", "query_type", "top1_type",
        "query_view", "top1_view",
        "same_view", "same_type", "same_upper", "same_lower", "same_bg",
    ]
    with open(os.path.join(save_dir, "ccpg_error_cases.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=case_fieldnames)
        writer.writeheader()
        writer.writerows(official_cases)

    if global_rows is not None:
        with open(os.path.join(save_dir, "ccpg_global_nn_error_covariates.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(global_rows)
    if global_cases is not None:
        with open(os.path.join(save_dir, "ccpg_global_nn_error_cases.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=case_fieldnames)
            writer.writeheader()
            writer.writerows(global_cases)


def cross_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    '''More details can be found: More details can be found in 
        [A Comprehensive Study on the Evaluation of Silhouette-based Gait Recognition]().
    '''
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-01'], 'BG': ['bg-01'], 'CL': ['cl-01']},
                      'OUMVLP': {'NM': ['00']}}

    gallery_seq_dict = {'CASIA-B': ['nm-02', 'bg-02', 'cl-02'],
                        'OUMVLP': ['01']}

    msg_mgr = get_msg_mgr()
    acc = {}
    mean_ap = {}
    view_list = sorted(np.unique(view))
    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros(len(view_list)) - 1.
        mean_ap[type_] = np.zeros(len(view_list)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset])
            gallery_y = label[gseq_mask]
            gallery_x = feature[gseq_mask, :]
            dist = cuda_dist(probe_x, gallery_x, metric)
            eval_results = compute_ACC_mAP(
                dist.cpu().numpy(), probe_y, gallery_y, view[pseq_mask], view[gseq_mask])
            acc[type_][v1] = np.round(eval_results[0] * 100, 2)
            mean_ap[type_][v1] = np.round(eval_results[1] * 100, 2)

    result_dict = {}
    msg_mgr.log_info(
        '===Cross View Gallery Evaluation (Excluded identical-view cases)===')
    out_acc_str = "========= Rank@1 Acc =========\n"
    out_map_str = "============= mAP ============\n"
    for type_ in probe_seq_dict[dataset].keys():
        avg_acc = np.mean(acc[type_])
        avg_map = np.mean(mean_ap[type_])
        result_dict[f'scalar/test_accuracy/{type_}-Rank@1'] = avg_acc
        result_dict[f'scalar/test_accuracy/{type_}-mAP'] = avg_map
        out_acc_str += f"{type_}:\t{acc[type_]}, mean: {avg_acc:.2f}%\n"
        out_map_str += f"{type_}:\t{mean_ap[type_]}, mean: {avg_map:.2f}%\n"
    # msg_mgr.log_info(f'========= Rank@1 Acc =========')
    msg_mgr.log_info(f'{out_acc_str}')
    # msg_mgr.log_info(f'========= mAP =========')
    msg_mgr.log_info(f'{out_map_str}')
    return result_dict



def single_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},
                      'OUMVLP': {'NM': ['00']},
                      'CASIA-E': {'NM': ['H-scene2-nm-1', 'H-scene2-nm-2', 'L-scene2-nm-1', 'L-scene2-nm-2', 'H-scene3-nm-1', 'H-scene3-nm-2', 'L-scene3-nm-1', 'L-scene3-nm-2', 'H-scene3_s-nm-1', 'H-scene3_s-nm-2', 'L-scene3_s-nm-1', 'L-scene3_s-nm-2', ],
                                  'BG': ['H-scene2-bg-1', 'H-scene2-bg-2', 'L-scene2-bg-1', 'L-scene2-bg-2', 'H-scene3-bg-1', 'H-scene3-bg-2', 'L-scene3-bg-1', 'L-scene3-bg-2', 'H-scene3_s-bg-1', 'H-scene3_s-bg-2', 'L-scene3_s-bg-1', 'L-scene3_s-bg-2'],
                                  'CL': ['H-scene2-cl-1', 'H-scene2-cl-2', 'L-scene2-cl-1', 'L-scene2-cl-2', 'H-scene3-cl-1', 'H-scene3-cl-2', 'L-scene3-cl-1', 'L-scene3-cl-2', 'H-scene3_s-cl-1', 'H-scene3_s-cl-2', 'L-scene3_s-cl-1', 'L-scene3_s-cl-2']
                                  },
                      'SUSTech1K': {'Normal': ['01-nm'], 'Bag': ['bg'], 'Clothing': ['cl'], 'Carrying':['cr'], 'Umberalla': ['ub'], 'Uniform': ['uf'], 'Occlusion': ['oc'],'Night': ['nt'], 'Overall': ['01','02','03','04']}
                      }
    gallery_seq_dict = {'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                        'OUMVLP': ['01'],
                        'CASIA-E': ['H-scene1-nm-1', 'H-scene1-nm-2', 'L-scene1-nm-1', 'L-scene1-nm-2'],
                        'SUSTech1K': ['00-nm'],}
    msg_mgr = get_msg_mgr()
    acc = {}
    view_list = sorted(np.unique(view))
    num_rank = 1
    if dataset == 'CASIA-E':
        view_list.remove("270")
    if dataset == 'SUSTech1K':
        num_rank = 5 
    view_num = len(view_list)

    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros((view_num, view_num, num_rank)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            pseq_mask = pseq_mask if 'SUSTech1K' not in dataset   else np.any(np.asarray(
                        [np.char.find(seq_type, probe)>=0 for probe in probe_seq]), axis=0
                            ) & np.isin(view, probe_view) # For SUSTech1K only
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset]) & np.isin(
                    view, [gallery_view])
                gseq_mask = gseq_mask if 'SUSTech1K' not in dataset  else np.any(np.asarray(
                            [np.char.find(seq_type, gallery)>=0 for gallery in gallery_seq_dict[dataset]]), axis=0
                                ) & np.isin(view, [gallery_view]) # For SUSTech1K only
                gallery_y = label[gseq_mask]
                gallery_x = feature[gseq_mask, :]
                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                acc[type_][v1, v2, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                                     0) * 100 / dist.shape[0], 2)

    result_dict = {}
    out_str = ""
    for rank in range(num_rank):
        out_str = ""
        for type_ in probe_seq_dict[dataset].keys():
            sub_acc = de_diag(acc[type_][:,:,rank], each_angle=True)
            if rank == 0:
                result_dict[f'scalar/test_accuracy/{type_}@R{rank+1}'] = np.mean(sub_acc)
            out_str += f"{type_}@R{rank+1}: {np.mean(sub_acc):.2f}%\t"
        msg_mgr.log_info("Indoor Rank-%d excl-identical: %s" % (rank + 1, out_str.strip()))
    return result_dict


def evaluate_indoor_dataset(data, dataset, metric='euc', cross_view_gallery=False):
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)

    if dataset not in ('CASIA-B', 'OUMVLP', 'CASIA-E', 'SUSTech1K'):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    if cross_view_gallery:
        return cross_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)
    else:
        return single_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)


def evaluate_real_scene(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)

    gallery_seq_type = {'0001-1000': ['1', '2'],
                        "HID2021": ['0'], '0001-1000-test': ['0'],
                        'GREW': ['01'], 'TTG-200': ['1']}
    probe_seq_type = {'0001-1000': ['3', '4', '5', '6'],
                      "HID2021": ['1'], '0001-1000-test': ['1'],
                      'GREW': ['02'], 'TTG-200': ['2', '3', '4', '5', '6']}

    num_rank = 20
    acc = np.zeros([num_rank]) - 1.
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = label[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
    acc = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                          0) * 100 / dist.shape[0], 2)
    msg_mgr.log_info('==Rank-1==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[0])))
    msg_mgr.log_info('==Rank-5==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[4])))
    msg_mgr.log_info('==Rank-10==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[9])))
    msg_mgr.log_info('==Rank-20==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[19])))
    return {"scalar/test_accuracy/Rank-1": np.mean(acc[0]), "scalar/test_accuracy/Rank-5": np.mean(acc[4])}


def GREW_submission(data, dataset, metric='euc'):
    get_msg_mgr().log_info("Evaluating GREW")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)
    gallery_seq_type = {'GREW': ['01', '02']}
    probe_seq_type = {'GREW': ['03']}
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = view[pseq_mask]

    num_rank = 20
    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()

    save_path = os.path.join(
        "GREW_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("GREW_result")
    with open(save_path, "w") as f:
        f.write("videoId,rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8,rank9,rank10,rank11,rank12,rank13,rank14,rank15,rank16,rank17,rank18,rank19,rank20\n")
        for i in range(len(idx)):
            r_format = [int(idx) for idx in gallery_y[idx[i, 0:num_rank]]]
            output_row = '{}'+',{}'*num_rank+'\n'
            f.write(output_row.format(probe_y[i], *r_format))
        print("GREW result saved to {}/{}".format(os.getcwd(), save_path))
    return


def HID_submission(data, dataset, rerank=True, metric='euc'):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating HID")
    feature, label, seq_type = data['embeddings'], data['labels'], data['views']
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]
    if rerank:
        feat = np.concatenate([probe_x, gallery_x])
        dist = cuda_dist(feat, feat, metric).cpu().numpy()
        msg_mgr.log_info("Starting Re-ranking")
        re_rank = re_ranking(
            dist, probe_x.shape[0], k1=6, k2=6, lambda_value=0.3)
        idx = np.argsort(re_rank, axis=1)
    else:
        dist = cuda_dist(probe_x, gallery_x, metric)
        idx = dist.cpu().sort(1)[1].numpy()

    save_path = os.path.join(
        "HID_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("HID_result")
    with open(save_path, "w") as f:
        f.write("videoID,label\n")
        for i in range(len(idx)):
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))
        print("HID result saved to {}/{}".format(os.getcwd(), save_path))
    return


def evaluate_segmentation(data, dataset):
    labels = data['mask']
    pred = data['pred']
    miou = mean_iou(pred, labels)
    get_msg_mgr().log_info('mIOU: %.3f' % (miou.mean()))
    return {"scalar/test_accuracy/mIOU": miou}


def evaluate_CCPG(data, dataset, metric='euc', log_detailed_rank=False, save_path=None,
                  analyze_feature_space=False, save_rank1_cases=False):
    msg_mgr = get_msg_mgr()

    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']

    label = np.array(label)
    seq_type_np = np.asarray(seq_type)
    for i in range(len(view)):
        view[i] = view[i].split("_")[0]
    view_np = np.array(view)
    view_list = list(set(view))
    view_list.sort()

    view_num = len(view_list)

    probe_seq_dict = {'CCPG': [["U0_D0_BG", "U0_D0"], [
        "U3_D3"], ["U1_D0"], ["U0_D0_BG"]]}

    gallery_seq_dict = {
        'CCPG': [["U1_D1", "U2_D2", "U3_D3"], ["U0_D3"], ["U1_D1"], ["U0_D0"]]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 100 if log_detailed_rank else 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                   view_num, view_num, num_rank]) - 1.

    setting_names = ['CL', 'UP', 'DN', 'BG']
    ap_save = []
    cmc_save = []
    cmc_full_save = []
    minp = []
    error_covariate_rows = []
    error_covariate_cases = []
    positive_rank_rows = []
    positive_rank_query_rows = []
    rank1_case_rows = []
    identity_margin_rows = []
    identity_margin_query_rows = []
    identity_margin_cases = []
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        gseq_mask = np.isin(seq_type, gallery_seq)
        gallery_indices = np.flatnonzero(gseq_mask)
        gallery_x = feature[gseq_mask, :]
        # print("gallery_x", gallery_x.shape)
        gallery_y = label[gseq_mask]
        gallery_view = view_np[gseq_mask]
        gallery_type = seq_type_np[gseq_mask]

        pseq_mask = np.isin(seq_type, probe_seq)
        probe_indices = np.flatnonzero(pseq_mask)
        probe_x = feature[pseq_mask, :]
        probe_y = label[pseq_mask]
        probe_view = view_np[pseq_mask]
        probe_type = seq_type_np[pseq_mask]

        distmat = cuda_dist(probe_x, gallery_x, metric).cpu().numpy()
        # cmc, ap = evaluate(distmat, probe_y, gallery_y, probe_view, gallery_view)
        cmc, ap, inp = evaluate_many(
            distmat, probe_y, gallery_y, probe_view, gallery_view)
        ap_save.append(ap)
        cmc_save.append(cmc[0])
        cmc_full_save.append(cmc)
        minp.append(inp)
        cov_row, cov_cases = _ccpg_rank1_error_covariates(
            distmat, probe_y, gallery_y, probe_view, gallery_view,
            probe_type, gallery_type, setting_names[p])
        error_covariate_rows.append(cov_row)
        error_covariate_cases.extend(cov_cases)
        rank_row, rank_query_rows = _ccpg_positive_rank_diagnostics(
            distmat, probe_y, gallery_y, probe_view, gallery_view,
            setting_names[p], probe_type)
        positive_rank_rows.append(rank_row)
        positive_rank_query_rows.extend(rank_query_rows)
        rank1_case_rows.extend(_ccpg_rank1_cases(
            distmat, probe_y, gallery_y, probe_view, gallery_view,
            probe_type, gallery_type, probe_indices, gallery_indices,
            setting_names[p]))
        margin_row, margin_query_rows, margin_cases = _ccpg_identity_margin_diagnostics(
            distmat, probe_y, gallery_y, probe_view, gallery_view,
            probe_type, gallery_type, setting_names[p])
        identity_margin_rows.append(margin_row)
        identity_margin_query_rows.extend(margin_query_rows)
        identity_margin_cases.extend(margin_cases)

    # print(ap_save, cmc_save)

    msg_mgr.log_info('CCPG ReID Rank-1 excl-identical: CL %.3f | UP %.3f | DN %.3f | BG %.3f' % (
        cmc_save[0]*100, cmc_save[1]*100, cmc_save[2]*100, cmc_save[3]*100))

    msg_mgr.log_info('CCPG ReID mAP excl-identical: CL %.3f | UP %.3f | DN %.3f | BG %.3f' % (
        ap_save[0]*100, ap_save[1]*100, ap_save[2]*100, ap_save[3]*100))

    msg_mgr.log_info('CCPG ReID mINP excl-identical: CL %.3f | UP %.3f | DN %.3f | BG %.3f' %
                     (minp[0]*100, minp[1]*100, minp[2]*100, minp[3]*100))

    if analyze_feature_space or save_rank1_cases:
        try:
            analysis_name = os.path.basename(save_path) if save_path else 'CCPG'
            analysis_dir = os.path.join(
                'analysis_outputs', 'ccpg_protocol', analysis_name)
            mkdir(analysis_dir)
            if analyze_feature_space:
                feature_vis = np.asarray(feature)
                if feature_vis.ndim == 3:
                    feature_vis = feature_vis.mean(axis=2)
                np.savez_compressed(
                    os.path.join(analysis_dir, "ccpg_embedding_vis_features.npz"),
                    features=feature_vis.astype(np.float32),
                    labels=np.asarray(label).astype(str),
                    types=seq_type_np.astype(str),
                    views=view_np.astype(str))
                cov_rows = error_covariate_rows + [
                    _ccpg_error_covariate_overall(error_covariate_rows)]
                global_rows, global_cases = _ccpg_global_nn_error_covariates(
                    feature, label, seq_type_np, view_np, metric)
                _write_ccpg_error_covariates(
                    analysis_dir, cov_rows, error_covariate_cases,
                    global_rows, global_cases)
                rank_rows = positive_rank_rows + [
                    _ccpg_positive_rank_summary("OVERALL", positive_rank_query_rows)]
                _write_ccpg_positive_rank_diagnostics(
                    analysis_dir, rank_rows, positive_rank_query_rows)
                margin_rows = identity_margin_rows + [
                    _ccpg_identity_margin_summary("OVERALL", identity_margin_query_rows)]
                _write_ccpg_identity_margin_diagnostics(
                    analysis_dir, margin_rows, identity_margin_cases)
                overall_cov = cov_rows[-1]
                global_cov = global_rows[0]
                overall_rank = rank_rows[-1]
                overall_margin = margin_rows[-1]
                msg_mgr.log_info(
                    "=== [CCPG protocol] rank-1 error covariates (official split) ===")
                msg_mgr.log_info(
                    "OVERALL: #err=%d, same_view=%.4f, view_lift=%.4f, "
                    "same_upper=%.4f, same_lower=%.4f, same_bg=%.4f" % (
                        overall_cov["n_error"],
                        overall_cov["same_view_among_errors"],
                        overall_cov["view_bias_lift"],
                        overall_cov["same_upper_among_errors"],
                        overall_cov["same_lower_among_errors"],
                        overall_cov["same_bg_among_errors"]))
                msg_mgr.log_info(
                    "=== [CCPG protocol] global NN error covariates (self excluded) ===")
                msg_mgr.log_info(
                    "OVERALL: #err=%d, same_view=%.4f, view_lift=%.4f, "
                    "same_type=%.4f, type_lift=%.4f" % (
                        global_cov["n_error"],
                        global_cov["same_view_among_errors"],
                        global_cov["view_bias_lift"],
                        global_cov["same_type_among_errors"],
                        global_cov["type_bias_lift"]))
                msg_mgr.log_info(
                    "=== [CCPG protocol] positive-rank diagnostics (official split) ===")
                msg_mgr.log_info(
                    "OVERALL: mAP=%.4f, mINP=%.4f, last_pos_rank_p90=%.2f, "
                    "coverage@50=%.4f, all_pos_top50=%.4f" % (
                        overall_rank["mAP"],
                        overall_rank["mINP"],
                        overall_rank["last_positive_rank_p90"],
                        overall_rank["positive_coverage_at_50"],
                        overall_rank["all_positives_in_top50_frac"]))
                msg_mgr.log_info(
                    "=== [CCPG protocol] identity-margin diagnostics (official split) ===")
                msg_mgr.log_info(
                    "OVERALL: violation=%.4f, margin_mean=%.4f, margin_p10=%.4f, "
                    "impostors_before_pos=%.4f" % (
                        overall_margin["violation_rate"],
                        overall_margin["margin_mean"],
                        overall_margin["margin_p10"],
                        overall_margin["impostors_before_first_positive_mean"]))
                msg_mgr.log_info(
                    "[CCPG protocol] error covariate analysis saved to %s" %
                    os.path.abspath(analysis_dir))
            if save_rank1_cases:
                _write_ccpg_rank1_cases(analysis_dir, rank1_case_rows)
                msg_mgr.log_info(
                    "[CCPG protocol] rank-1 cases saved to %s" %
                    os.path.abspath(os.path.join(analysis_dir, "ccpg_rank1_cases.csv")))
        except Exception as e:
            msg_mgr.log_warning(
                "[CCPG protocol] error covariate analysis failed: %s" % repr(e))

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        for (v1, probe_view) in enumerate(view_list):
            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                    view, [gallery_view])
                gallery_x = feature[gseq_mask, :]
                gallery_y = label[gseq_mask]

                pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                    view, [probe_view])
                probe_x = feature[pseq_mask, :]
                probe_y = label[pseq_mask]

                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.sort(1)[1].cpu().numpy()
                effective_rank = min(num_rank, dist.shape[1])
                rank_acc = np.round(
                    np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:effective_rank]], 1) > 0,
                           0) * 100 / dist.shape[0], 2)
                acc[p, v1, v2, :effective_rank] = rank_acc
                if effective_rank < num_rank:
                    acc[p, v1, v2, effective_rank:] = rank_acc[-1]

    result_dict = {}
    for i in range(1):
        msg_mgr.log_info('CCPG Gait Rank-%d excl-identical: CL %.3f | UP %.3f | DN %.3f | BG %.3f' % (
            i + 1,
            de_diag(acc[0, :, :, i]),
            de_diag(acc[1, :, :, i]),
            de_diag(acc[2, :, :, i]),
            de_diag(acc[3, :, :, i])))
    result_dict["scalar/test_accuracy/CL"] = acc[0, :, :, i]
    result_dict["scalar/test_accuracy/UP"] = acc[1, :, :, i]
    result_dict["scalar/test_accuracy/DN"] = acc[2, :, :, i]
    result_dict["scalar/test_accuracy/BG"] = acc[3, :, :, i]
    if log_detailed_rank and save_path:
        import csv
        ranks_dir = os.path.join(save_path, "ranks")
        mkdir(ranks_dir)
        timestamp = strftime('%Y-%m-%d-%H-%M-%S', localtime())

        reid_csv_path = os.path.join(ranks_dir, 'ccpg_reid_cmc_{}.csv'.format(timestamp))
        max_reid_rank = min(100, min(len(c) for c in cmc_full_save))
        with open(reid_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['rank'] + setting_names + ['mean'])
            for r in range(max_reid_rank):
                vals = [cmc_full_save[i][r] * 100 for i in range(len(setting_names))]
                row = [r + 1] + ['{:.4f}'.format(v) for v in vals] + ['{:.4f}'.format(np.mean(vals))]
                writer.writerow(row)
        msg_mgr.log_info('ReID protocol CMC ranks (1-{}) saved to {}'.format(max_reid_rank, reid_csv_path))

        gait_csv_path = os.path.join(ranks_dir, 'ccpg_gait_cmc_{}.csv'.format(timestamp))
        with open(gait_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['rank'] + setting_names + ['mean'])
            for r in range(num_rank):
                vals = [de_diag(acc[i, :, :, r]) for i in range(len(setting_names))]
                row = [r + 1] + ['{:.4f}'.format(v) for v in vals] + ['{:.4f}'.format(np.mean(vals))]
                writer.writerow(row)
        msg_mgr.log_info('Gait protocol CMC ranks (1-{}) saved to {}'.format(num_rank, gait_csv_path))

    # Optional CCPG-protocol feature-space diagnostics (distance entanglement
    # table + rank-1 error -> same-view breakdown). Opt-in via the evaluator
    # config flag `analyze_feature_space: true`; default off so other runs are
    # unaffected. Runs on the SAME feature/label/outfit/view arrays + protocol
    # splits used above, hence fully consistent with the reported metric.
    if analyze_feature_space:
        try:
            from .ccpg_analysis import run_ccpg_feature_analysis
            analysis_name = os.path.basename(save_path) if save_path else 'CCPG'
            analysis_dir = os.path.join(
                'analysis_outputs', 'ccpg_protocol', analysis_name)
            run_ccpg_feature_analysis(
                feature=feature, label=label, seq_type=seq_type, view_np=view_np,
                probe_seq_dict=probe_seq_dict[dataset],
                gallery_seq_dict=gallery_seq_dict[dataset],
                setting_names=setting_names, metric=metric,
                save_dir=analysis_dir)
        except Exception as e:  # never break the standard evaluation
            msg_mgr.log_warning(
                "[CCPG protocol] feature-space analysis failed: %s" % repr(e))

    return result_dict

def evaluate_scoliosis(data, dataset, metric='euc'):

    msg_mgr = get_msg_mgr()
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    logits = np.array(data['embeddings'])
    labels = data['types']
    
    # Label mapping: negative->0, neutral->1, positive->2  
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    true_ids = np.array([label_map[status] for status in labels])
    
    pred_ids = np.argmax(logits.mean(-1), axis=-1)
    
    # Calculate evaluation metrics
    # Total Accuracy: proportion of correctly predicted samples among all samples
    accuracy = accuracy_score(true_ids, pred_ids)
    
    # Macro-average Precision: average of precision scores for each class
    precision = precision_score(true_ids, pred_ids, average='macro', zero_division=0)
    
    # Macro-average Recall: average of recall scores for each class  
    recall = recall_score(true_ids, pred_ids, average='macro', zero_division=0)
    
    # Macro-average F1: average of F1 scores for each class
    f1 = f1_score(true_ids, pred_ids, average='macro', zero_division=0)
    
    # Confusion matrix (for debugging)
    # cm = confusion_matrix(true_ids, pred_ids, labels=[0, 1, 2])
    # class_names = ['Negative', 'Neutral', 'Positive']
    
    # Print results
    msg_mgr.log_info(f"Total Accuracy: {accuracy*100:.2f}%")
    msg_mgr.log_info(f"Macro-avg Precision: {precision*100:.2f}%") 
    msg_mgr.log_info(f"Macro-avg Recall: {recall*100:.2f}%")
    msg_mgr.log_info(f"Macro-avg F1 Score: {f1*100:.2f}%")
    
    return {
        "scalar/test_accuracy/": accuracy,
        "scalar/test_precision/": precision, 
        "scalar/test_recall/": recall,
        "scalar/test_f1/": f1
    }
