import csv
import math
import os.path as osp
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as tordata
from collections import defaultdict

from itertools import combinations

from utils import get_msg_mgr


# The modules below back optional experiments (Partial-GT, pseudo conditions,
# nuisance corruption). When a module is absent, its names are
# replaced by stubs that raise only if the feature is actually used, so plain
# NCB / triplet training still runs.
def _missing(module, name):
    def _raise(*args, **kwargs):
        raise ImportError(
            "data.%s is not available, so %s cannot be used. Restore "
            "opengait/data/%s.py to enable this feature." % (module, name, module))
    return _raise


try:
    from .partial_annotation import (MASKED_VIEW, MASKED_VIEW_ID, annotation_stats,
                                     build_annotation_masks, format_stats,
                                     manifest_filename, sequence_key, write_manifest)
except ImportError:
    # Sentinel that never equals a real view label; maps to "no view" (-1).
    MASKED_VIEW = '__masked_view__'
    MASKED_VIEW_ID = -1

    def sequence_key(identity, condition, view):
        # In-memory key only; offline artefacts written with the original
        # partial_annotation.sequence_key may use a different format.
        return '%s/%s/%s' % (identity, condition, view)

    annotation_stats = _missing('partial_annotation', 'annotation_stats')
    build_annotation_masks = _missing('partial_annotation', 'build_annotation_masks')
    format_stats = _missing('partial_annotation', 'format_stats')
    manifest_filename = _missing('partial_annotation', 'manifest_filename')
    write_manifest = _missing('partial_annotation', 'write_manifest')

try:
    from .pseudo_condition import (PseudoConditionBuilder, camera_center_features,
                                     load_pca_features)
except ImportError:
    PseudoConditionBuilder = _missing('pseudo_condition', 'PseudoConditionBuilder')
    camera_center_features = _missing('pseudo_condition', 'camera_center_features')
    load_pca_features = _missing('pseudo_condition', 'load_pca_features')

try:
    from .nuisance_corruption import (build_corrupted_metadata,
                                      format_corruption_audit)
except ImportError:
    build_corrupted_metadata = _missing('nuisance_corruption', 'build_corrupted_metadata')
    format_corruption_audit = _missing('nuisance_corruption', 'format_corruption_audit')


def _log(message):
    """Log to the training log if it is up, else stdout (self-tests, dry runs)."""
    msg_mgr = get_msg_mgr()
    if getattr(msg_mgr, 'logger', None) is not None:
        for line in str(message).splitlines():
            msg_mgr.log_info(line)
    else:
        print(message)


def _coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean()) if values.size else 0.0
    return float(values.std() / mean) if mean > 0 else 0.0


def _gini(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    total = float(values.sum())
    if not values.size or total <= 0:
        return 0.0
    positions = np.arange(1, len(values) + 1, dtype=np.float64)
    return float((2.0 * np.dot(positions, values) / (len(values) * total))
                 - (len(values) + 1.0) / len(values))

class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


def sync_random_sample_list(obj_list, k, common_choice=False):
    if common_choice:
        idx = random.choices(range(len(obj_list)), k=k) 
        idx = torch.tensor(idx)
    if len(obj_list) < k:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]


def load_pseudo_view_map(path):
    if not osp.isfile(path):
        raise FileNotFoundError("pseudo_view_file not found: %s" % path)
    required = {'identity', 'condition', 'view_raw', 'pseudo_view_cluster'}
    mapping = {}
    with open(path, 'r', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "pseudo_view_file %s is missing columns %s" % (path, sorted(missing)))
        for row in reader:
            key = (row['identity'], row['condition'], row['view_raw'])
            if key in mapping:
                raise ValueError(
                    "pseudo_view_file %s contains duplicate key %s" % (path, key))
            label = int(row['pseudo_view_cluster'])
            if label < -1:
                raise ValueError(
                    "pseudo_view_file %s has invalid cluster %d for %s; "
                    "only -1 (rejected) or non-negative labels are allowed"
                    % (path, label, key))
            mapping[key] = label
    if not mapping:
        raise ValueError("pseudo_view_file %s contains no rows" % path)
    return mapping


def load_pseudo_condition_map(path, column='pseudo_condition_local'):
    if not osp.isfile(path):
        raise FileNotFoundError("pseudo_condition.assignment_file not found: %s" % path)
    required = {'identity', 'condition', 'view_raw', column}
    mapping = {}
    with open(path, 'r', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "pseudo_condition.assignment_file %s is missing columns %s"
                % (path, sorted(missing)))
        for row in reader:
            key = sequence_key(row['identity'], row['condition'], row['view_raw'])
            if key in mapping:
                raise ValueError(
                    "pseudo_condition.assignment_file %s contains duplicate key %s"
                    % (path, key))
            label = int(row[column])
            if label not in (-1, 0, 1):
                raise ValueError(
                    "pseudo condition label for %s must be -1, 0 or 1, got %d"
                    % (key, label))
            mapping[key] = label
    if not mapping:
        raise ValueError("pseudo_condition.assignment_file %s contains no rows" % path)
    return mapping


def load_resolved_metadata_map(path):
    if not osp.isfile(path):
        raise FileNotFoundError("resolved_metadata_path not found: %s" % path)
    required = {
        'identity', 'sequence', 'pseudo_view', 'pseudo_clothes',
        'view_gt_available', 'clothes_gt_available',
        'final_view', 'final_clothes',
    }
    forbidden = {'gt_view', 'view_gt', 'gt_clothes', 'clothes_gt',
                 'condition_gt', 'gt_condition'}
    mapping = {}
    n_view_gt = 0
    n_clothes_gt = 0
    with open(path, 'r', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(
                "resolved_metadata_path %s is missing columns %s"
                % (path, sorted(missing)))
        leaked = forbidden & fields
        if leaked:
            raise ValueError(
                "training metadata %s contains evaluation-only GT columns %s"
                % (path, sorted(leaked)))
        for row in reader:
            key = row['sequence']
            if key in mapping:
                raise ValueError(
                    "resolved_metadata_path %s contains duplicate sequence %s"
                    % (path, key))
            if not row['final_view'] or not row['final_clothes']:
                raise ValueError(
                    "resolved metadata has an empty final label for %s" % key)
            view_gt_available = bool(int(row['view_gt_available']))
            clothes_gt_available = bool(int(row['clothes_gt_available']))
            n_view_gt += int(view_gt_available)
            n_clothes_gt += int(clothes_gt_available)
            mapping[key] = {
                'identity': row['identity'],
                'final_view': row['final_view'],
                'final_clothes': row['final_clothes'],
                'view_gt_available': view_gt_available,
                'clothes_gt_available': clothes_gt_available,
            }
    if not mapping:
        raise ValueError("resolved_metadata_path %s contains no rows" % path)
    return mapping, {
        'n_sequences': len(mapping),
        'n_view_gt': n_view_gt,
        'n_clothes_gt': n_clothes_gt,
        'view_gt_ratio': n_view_gt / len(mapping),
        'clothes_gt_ratio': n_clothes_gt / len(mapping),
    }


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)


class CommonSampler(tordata.sampler.Sampler):
    def __init__(self,dataset,batch_size,batch_shuffle):

        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size,int)==False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle
        
        self.world_size = dist.get_world_size()
        if self.batch_size % self.world_size !=0:
            raise ValueError("World size ({}) is not divisble by batch_size ({})".format(
                self.world_size, batch_size))
        self.rank = dist.get_rank() 
    
    def __iter__(self):
        while True:
            indices_list = list(range(self.size))
            sample_indices = sync_random_sample_list(
                    indices_list, self.batch_size, common_choice=True)
            total_batch_size =  self.batch_size
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]
            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

# **************** For GaitSSB ****************
import random
class BilateralSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.dataset_length = len(self.dataset)
        self.total_indices = list(range(self.dataset_length))

    def __iter__(self):
        random.shuffle(self.total_indices)
        count = 0
        batch_size = self.batch_size[0] * self.batch_size[1]
        while True:
            if (count + 1) * batch_size >= self.dataset_length:
                count = 0
                random.shuffle(self.total_indices)

            sampled_indices = self.total_indices[count*batch_size:(count+1)*batch_size]
            sampled_indices = sync_random_sample_list(sampled_indices, len(sampled_indices))

            total_size = int(math.ceil(batch_size / self.world_size)) * self.world_size
            sampled_indices += sampled_indices[:(batch_size - len(sampled_indices))]

            sampled_indices = sampled_indices[self.rank:total_size:self.world_size]
            count += 1

            yield sampled_indices * 2

    def __len__(self):
        return len(self.dataset)



# **************** For NCB Sampling ****************
class NCBSampler(tordata.sampler.Sampler):
    STRICT_POLICY = 'strict_resample_then_pk'
    PARTIAL_CONTROLLED_POLICY = 'partial_controlled_then_pk'
    # CCPG has 10 cameras -> 210 sets of 4, enumerated exhaustively. The cap only
    # matters for label spaces large enough to make that intractable.
    MAX_VIEW_SET_ENUMERATION = 20000

    def __init__(self, dataset, batch_size, batch_shuffle=False,
                 outfit_key_field='type', min_views_per_outfit=2,
                 V_shared=4, per_view_shots=4,
                 select_per_view=None, n_outfit_blocks=1,
                 selection_policy='uniform', deterministic_selection=False,
                 fallback_policy='coverage_top',
                 max_resample_trials=20, ensure_min_shared_coverage=None,
                 view_selection_mode='uniform',
                 view_source=None,
                 pseudo_view_file=None, apply_pseudo_view_to_dataset=True,
                 pseudo_condition=None,
                 partial_annotation=None,
                 metadata_mode=None, gt_view_ratio=None,
                 gt_clothes_ratio=None, partial_seed=None,
                 resolved_metadata_path=None,
                 nuisance_corruption=None,
                 log_batch_stats=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError("batch_size should be (P x K) not {}".format(batch_size))

        # Snapshot the original GT metadata before any relabelling below rewrites
        # seqs_info[2]; offline artefacts (pseudo-view csv, GEI features) are keyed
        # on it, and it is the only way back to the raw view once clusters are in.
        self.sequence_keys = [sequence_key(info[0], info[1], info[2])
                              for info in dataset.seqs_info]

        P, K = self.batch_size
        self.batch_shuffle = batch_shuffle
        self.outfit_key_field = outfit_key_field
        self.min_views_per_outfit = min_views_per_outfit
        self.V_shared = V_shared
        self.per_view_shots = per_view_shots
        self.select_per_view = select_per_view if select_per_view is not None else per_view_shots
        self.n_outfit_blocks = n_outfit_blocks
        self.selection_policy = selection_policy
        self.deterministic_selection = deterministic_selection
        self.fallback_policy = fallback_policy
        self.max_resample_trials = max_resample_trials
        self.ensure_min_shared_coverage = ensure_min_shared_coverage if ensure_min_shared_coverage is not None else max(1, int(0.75 * P))
        self.view_selection_mode = view_selection_mode
        self.log_batch_stats = log_batch_stats
        self.iteration = 0
        self.last_block_plan = []
        self.last_shared_views = []

        self.strict_mode = (fallback_policy == self.STRICT_POLICY)
        self.partial_controlled_mode = (
            fallback_policy == self.PARTIAL_CONTROLLED_POLICY)
        if self.view_selection_mode not in (
                'uniform', 'inverse_exposure', 'soft_exposure_regularization'):
            raise ValueError(
                "view_selection_mode must be 'uniform', 'inverse_exposure' or "
                "'soft_exposure_regularization', got "
                f"{self.view_selection_mode}")

        if self.select_per_view > self.per_view_shots:
            raise ValueError(
                f"select_per_view ({self.select_per_view}) must be <= "
                f"per_view_shots ({self.per_view_shots})")

        # Validate K = n_outfit_blocks * V_shared * select_per_view
        expected_K = n_outfit_blocks * V_shared * self.select_per_view
        if K != expected_K:
            if self.strict_mode or self.partial_controlled_mode:
                raise ValueError(
                    f"{fallback_policy} must produce exactly P*K samples, but "
                    f"K={K} != n_blocks({n_outfit_blocks})*V({V_shared})"
                    f"*sel({self.select_per_view})={expected_K}.")
            print(f"[NCBSampler] WARNING: K={K} but "
                  f"n_blocks({n_outfit_blocks})*V({V_shared})*sel({self.select_per_view})={expected_K}. "
                  f"Will use flexible allocation.")
            self.flexible_K = True
        else:
            self.flexible_K = False

        # Tracking stats for shared views enforcement
        self.total_samples = 0
        self.fallback_samples = 0
        self.batch_stats_history = []
        self._reset_strict_stats()

        self.world_size = dist.get_world_size()
        if (P * K) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, P, K))
        self.rank = dist.get_rank()

        if resolved_metadata_path and (pseudo_view_file or pseudo_condition
                                       or partial_annotation):
            raise ValueError(
                'resolved_metadata_path is already fully resolved and cannot be '
                'combined with pseudo_view_file, pseudo_condition, or '
                'partial_annotation')
        corruption_enabled = bool(
            nuisance_corruption and nuisance_corruption.get('enabled', False))
        if corruption_enabled and (resolved_metadata_path or pseudo_view_file
                                   or pseudo_condition or partial_annotation):
            raise ValueError(
                'nuisance_corruption starts from GT train metadata and cannot be '
                'combined with resolved, pseudo, or partial metadata options')
        if resolved_metadata_path and metadata_mode != 'partial_unknown':
            raise ValueError(
                "resolved_metadata_path requires metadata_mode='partial_unknown'")
        if metadata_mode and metadata_mode != 'partial_unknown':
            raise ValueError("unsupported metadata_mode %r" % metadata_mode)

        # view_source=None keeps the legacy rule (pseudo iff a file was given) so
        # existing configs are untouched; resolved metadata owns both axes.
        inferred_source = ('resolved' if resolved_metadata_path else
                           ('pseudo' if pseudo_view_file else 'gt'))
        self.view_source = view_source or inferred_source
        if self.view_source not in ('gt', 'pseudo', 'resolved'):
            raise ValueError(
                "view_source must be 'gt', 'pseudo' or 'resolved', got %r"
                % view_source)
        if self.view_source == 'pseudo' and not pseudo_view_file:
            raise ValueError("view_source='pseudo' requires pseudo_view_file")
        if view_source == 'gt' and pseudo_view_file:
            raise ValueError("view_source='gt' conflicts with pseudo_view_file=%r"
                             % pseudo_view_file)
        if self.view_source == 'resolved' and not resolved_metadata_path:
            raise ValueError(
                "view_source='resolved' requires resolved_metadata_path")

        self.pseudo_view_map = None
        self.pseudo_view_valid = None
        if pseudo_view_file:
            self.pseudo_view_map = load_pseudo_view_map(pseudo_view_file)
            if apply_pseudo_view_to_dataset:
                self._apply_pseudo_view_labels()

        self.resolved_metadata = None
        self.resolved_metadata_stats = None
        if resolved_metadata_path:
            (self.resolved_metadata,
             self.resolved_metadata_stats) = load_resolved_metadata_map(
                 resolved_metadata_path)
            self._apply_resolved_metadata_labels(
                gt_view_ratio=gt_view_ratio,
                gt_clothes_ratio=gt_clothes_ratio,
                partial_seed=partial_seed)

        self.nuisance_metadata = None
        self.nuisance_corruption_stats = None
        if corruption_enabled:
            self._apply_nuisance_corruption(nuisance_corruption)

        self.view_valid = None
        self.condition_valid = None
        self.annotation_stats = None
        self.partial_enabled = False
        self.joint_identities = []
        self.complement_identities = []
        self.joint_identity_ratio = 1.0
        if partial_annotation and partial_annotation.get('enabled', False):
            self._apply_partial_annotation(partial_annotation)

        self.pseudo_condition_enabled = bool(
            pseudo_condition and pseudo_condition.get('enabled', False))
        self.pseudo_condition_offline = False
        self.pseudo_condition_labels = None
        if self.pseudo_condition_enabled:
            self._setup_pseudo_condition(pseudo_condition)

        self._build_outfit_view_index()

        self.feasible_view_sets = {}
        if self.strict_mode and (self.partial_enabled or self.pseudo_condition_enabled):
            self._build_feasible_view_sets()
        if self.rank == 0 and self.log_batch_stats:
            self._log_statistics()

    def _reset_strict_stats(self):
        self.n_batches = 0
        self.n_perfect_intersection = 0
        self.n_resample_success = 0
        self.n_standard_pk = 0
        self.n_resample_trials = 0
        self.n_cell_draws = 0
        self.n_cell_draws_with_replacement = 0
        self.view_exposure = defaultdict(int)
        self.identity_exposure = defaultdict(int)
        self.n_scheduled_pk = 0
        self.last_batch_size = self.batch_size[0] * self.batch_size[1]
        # Why strict trials fail: too few distinct annotated conditions for some
        # identity, or no V_shared-wide exact view intersection.
        self.n_trials_no_distinct_conditions = 0
        self.n_trials_no_shared_views = 0
        self.best_intersection_size = 0
        self.n_partial_controlled_full = 0
        self.n_partial_controlled_partial = 0
        self.n_partial_controlled_pk = 0
        self.n_partial_controlled_slots = 0
        self.n_partial_controlled_pk_fill_slots = 0
        # Tier counts for the per-identity shared-viewpoint fallback.
        self.n_identity_full = 0
        self.n_identity_partial = 0
        self.n_identity_pk = 0

    def _apply_pseudo_view_labels(self):
        """Replace the train dataset view field with the offline cluster id.

        ``seqs_info[i][2]`` feeds both this sampler and ``BaseModel._get_view_indices``
        (which passes non-string views straight through), so rewriting it once keeps
        the batch construction and the view-match negative mining on the same label.
        """
        seqs_info = self.dataset.seqs_info
        unmapped = []
        valid = []
        for idx, seq_info in enumerate(seqs_info):
            key = (seq_info[0], seq_info[1], seq_info[2])
            if key not in self.pseudo_view_map:
                unmapped.append(key)
                valid.append(False)
                continue
            label = self.pseudo_view_map[key]
            # -1 is shared with BaseModel/TripletLoss as the invalid-view
            # sentinel.  It remains in the dataset for whole-batch PK fallback,
            # but _is_ncb_eligible excludes it from pseudo-view cells.
            seqs_info[idx][2] = label
            valid.append(label >= 0)
        if unmapped:
            raise ValueError(
                "pseudo_view_file covers %d/%d training sequences; missing e.g. %s. "
                "A partial mapping would silently mix cluster ids and camera ids."
                % (len(seqs_info) - len(unmapped), len(seqs_info), unmapped[:3]))
        self.pseudo_view_valid = np.asarray(valid, dtype=bool)
        # Keep the cached views in sync for anything that reads them later.
        if hasattr(self.dataset, 'views_list'):
            self.dataset.views_list = [info[2] for info in seqs_info]
            self.dataset.views_set = sorted(set(self.dataset.views_list), key=str)
        if self.rank == 0:
            clusters = {value for value in self.pseudo_view_map.values() if value >= 0}
            print("[NCBSampler] applied %d pseudo-view labels (%d retained, %d "
                  "clusters); rejected labels stay invalid for NCB/VM"
                  % (len(seqs_info), int(self.pseudo_view_valid.sum()), len(clusters)))

    def _apply_resolved_metadata_labels(self, gt_view_ratio=None,
                                        gt_clothes_ratio=None,
                                        partial_seed=None):
        seqs_info = self.dataset.seqs_info
        missing = [key for key in self.sequence_keys
                   if key not in self.resolved_metadata]
        extras = sorted(set(self.resolved_metadata) - set(self.sequence_keys))
        if missing or extras:
            raise ValueError(
                'resolved metadata/dataset key mismatch: covered %d/%d; '
                'missing e.g. %s; extra e.g. %s'
                % (len(self.sequence_keys) - len(missing), len(self.sequence_keys),
                   missing[:3], extras[:3]))

        for idx, (key, seq_info) in enumerate(zip(self.sequence_keys, seqs_info)):
            row = self.resolved_metadata[key]
            if str(seq_info[0]) != row['identity']:
                raise ValueError(
                    'resolved metadata identity mismatch for %s: %s != %s'
                    % (key, row['identity'], seq_info[0]))
            seqs_info[idx][1] = row['final_clothes']
            seqs_info[idx][2] = row['final_view']

        stats = self.resolved_metadata_stats
        tolerance = 1.0 / max(stats['n_sequences'], 1) + 1e-12
        for name, configured, actual in (
                ('gt_view_ratio', gt_view_ratio, stats['view_gt_ratio']),
                ('gt_clothes_ratio', gt_clothes_ratio,
                 stats['clothes_gt_ratio'])):
            if configured is not None and abs(float(configured) - actual) > 0.02 + tolerance:
                raise ValueError(
                    '%s=%s disagrees with resolved metadata coverage %.6f'
                    % (name, configured, actual))

        if hasattr(self.dataset, 'types_list'):
            self.dataset.types_list = [info[1] for info in seqs_info]
            self.dataset.types_set = sorted(set(self.dataset.types_list), key=str)
        if hasattr(self.dataset, 'views_list'):
            self.dataset.views_list = [info[2] for info in seqs_info]
            self.dataset.views_set = sorted(set(self.dataset.views_list), key=str)
        if self.rank == 0:
            _log('[NCBSampler] resolved metadata applied: %d sequences, '
                 'view GT %.4f, clothes GT %.4f, seed=%s; training reads only '
                 'final_view/final_clothes'
                 % (stats['n_sequences'], stats['view_gt_ratio'],
                    stats['clothes_gt_ratio'], partial_seed))

    def _apply_nuisance_corruption(self, cfg):
        """Corrupt train nuisance metadata once, before sampler indices exist."""
        records, stats = build_corrupted_metadata(
            self.dataset.seqs_info,
            view_rate=cfg.get('view_rate', 0.0),
            clothes_rate=cfg.get('clothes_rate', 0.0),
            seed=cfg.get('seed', 0),
            stratify_by=cfg.get('stratify_by', ['identity', 'view']))
        self.nuisance_metadata = records
        self.nuisance_corruption_stats = stats

        # At 0%, leave every seq_info object byte-for-byte untouched so this is
        # exactly the fully annotated baseline apart from audit bookkeeping.
        for record in records:
            index = record['index']
            if record['view_corrupted']:
                self.dataset.seqs_info[index][2] = record['corrupted_view']
            if record['clothes_corrupted']:
                self.dataset.seqs_info[index][1] = record['corrupted_clothes']

        if stats['n_view_corrupted'] and hasattr(self.dataset, 'views_list'):
            self.dataset.views_list = [info[2] for info in self.dataset.seqs_info]
            self.dataset.views_set = sorted(set(self.dataset.views_list), key=str)
        if stats['n_clothes_corrupted'] and hasattr(self.dataset, 'types_list'):
            self.dataset.types_list = [info[1] for info in self.dataset.seqs_info]
            self.dataset.types_set = sorted(set(self.dataset.types_list), key=str)

        if self.rank == 0:
            _log(format_corruption_audit(
                records, stats, seed=cfg.get('seed', 0),
                max_examples=cfg.get('max_examples', 8)))

    def _apply_partial_annotation(self, cfg):
        seqs_info = self.dataset.seqs_info
        records = []
        for seq_info in seqs_info:
            identity, condition, view_raw = seq_info[0], seq_info[1], seq_info[2]
            records.append({
                'sequence_key': sequence_key(identity, condition, view_raw),
                'identity': identity,
                'condition': condition,
                'view_raw': view_raw,
                'gt_view': self._normalize_view_label(view_raw),
            })

        view_valid, condition_valid = build_annotation_masks(
            records,
            view_ratio=cfg.get('view_ratio', 1.0),
            condition_ratio=cfg.get('condition_ratio', 1.0),
            seed=cfg.get('seed', 0),
            overlap_mode=cfg.get('overlap_mode', 'aligned'),
            mask_unit=cfg.get('mask_unit', 'identity'))

        for idx, ok in enumerate(view_valid):
            if not ok:
                seqs_info[idx][2] = MASKED_VIEW
        if hasattr(self.dataset, 'views_list'):
            self.dataset.views_list = [info[2] for info in seqs_info]
            self.dataset.views_set = sorted(set(self.dataset.views_list), key=str)

        self.view_valid = view_valid
        self.condition_valid = condition_valid
        self.partial_enabled = True
        self.annotation_stats = annotation_stats(records, view_valid, condition_valid, cfg)

        joint = {r['identity'] for r, v, c in zip(records, view_valid, condition_valid)
                 if v and c}
        every = {r['identity'] for r in records}
        self.joint_identities = sorted(joint)
        self.complement_identities = sorted(every - joint)
        self.joint_identity_ratio = len(joint) / max(len(every), 1)

        if self.rank == 0:
            _log(format_stats(self.annotation_stats))
            if cfg.get('save_manifest', False):
                path = osp.join(cfg.get('manifest_dir', './analysis/partial_annotation'),
                                manifest_filename(cfg))
                write_manifest(path, records, view_valid, condition_valid)
                _log('[PartialAnnotation] manifest -> %s' % path)

    def _setup_pseudo_condition(self, cfg):
        """Load frozen labels or legacy PCA-GEI features for condition blocks."""
        if self.n_outfit_blocks != 2:
            raise ValueError(
                'pseudo_condition builds exactly two blocks (A and B), so '
                'n_outfit_blocks must be 2, got %d' % self.n_outfit_blocks)
        if not self.strict_mode:
            raise ValueError('pseudo_condition requires fallback_policy=%s'
                             % self.STRICT_POLICY)

        assignment_file = cfg.get('assignment_file')
        if assignment_file:
            column = cfg.get('assignment_column', 'pseudo_condition_local')
            by_key = load_pseudo_condition_map(assignment_file, column)
            missing = [key for key in self.sequence_keys if key not in by_key]
            if missing:
                raise ValueError(
                    'pseudo_condition.assignment_file %s covers %d/%d training '
                    'sequences; missing e.g. %s'
                    % (assignment_file, len(self.sequence_keys) - len(missing),
                       len(self.sequence_keys), missing[:3]))
            self.pseudo_condition_labels = np.asarray(
                [by_key[key] for key in self.sequence_keys], dtype=np.int8)
            self.pseudo_condition_offline = True
            self.pseudo_condition_matching = 'offline_assignment'
            self.pseudo_condition_builder = None
            self.last_block_plan = []
            if self.rank == 0:
                counts = {label: int((self.pseudo_condition_labels == label).sum())
                          for label in (-1, 0, 1)}
                _log('[NCBSampler] frozen view-local pseudo-condition labels from '
                     '%s column=%s counts=%s; GT condition labels are not read'
                     % (assignment_file, column, counts))
            return

        feature_file = cfg.get('feature_file')
        if not feature_file:
            raise ValueError(
                'pseudo_condition.enabled requires assignment_file or feature_file')
        by_key = load_pca_features(feature_file)

        missing = [key for key in self.sequence_keys if key not in by_key]
        if missing:
            raise ValueError(
                'pseudo_condition.feature_file %s covers %d/%d training sequences; '
                'missing e.g. %s. Regenerate it with tools/gei_view_cluster_analysis.py '
                '--split train --pca-dim 50.'
                % (feature_file, len(self.sequence_keys) - len(missing),
                   len(self.sequence_keys), missing[:3]))

        matrix = np.stack([by_key[key] for key in self.sequence_keys]).astype(np.float64)
        self.pseudo_condition_matching = cfg.get(
            'matching_policy', 'joint_distinct_assignment')
        ratio_max = float(cfg.get('ratio_max', 0.90))
        if self.pseudo_condition_matching == 'paired_ratio':
            views = [self._normalize_view_label(info[2])
                     for info in self.dataset.seqs_info]
            matrix = camera_center_features(matrix, views)
        self.pseudo_condition_builder = PseudoConditionBuilder(
            matrix,
            reference_policy=cfg.get('reference_policy', 'max_farthest_pair'),
            matching_policy=self.pseudo_condition_matching,
            ratio_max=ratio_max)
        self.last_block_plan = []
        if self.rank == 0:
            extra = ''
            if self.pseudo_condition_matching == 'paired_ratio':
                extra = ', ratio_max=%.2f, features camera-centered' % ratio_max
            _log('[NCBSampler] pseudo-condition blocks from %s (%d sequences x %d dims), '
                 'reference=%s matching=%s%s; GT condition labels are not read'
                 % (feature_file, matrix.shape[0], matrix.shape[1],
                    cfg.get('reference_policy', 'max_farthest_pair'),
                    self.pseudo_condition_matching, extra))

    def _build_view_index(self):
        """id2view2indices[identity][view] = indices, ignoring the GT condition."""
        self.id2view2indices = defaultdict(lambda: defaultdict(list))
        self.id2view2condition2indices = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        for idx in range(len(self.dataset)):
            if not self._is_ncb_eligible(idx):
                continue
            seq_info = self.dataset.seqs_info[idx]
            identity = seq_info[0]
            view = self._normalize_view_label(seq_info[2])
            self.id2view2indices[identity][view].append(idx)
            if self.pseudo_condition_offline:
                side = int(self.pseudo_condition_labels[idx])
                self.id2view2condition2indices[identity][view][side].append(idx)

    def _is_ncb_eligible(self, idx):
        """NCB blocks need both GT annotations; PK fallback needs neither."""
        if self.pseudo_view_valid is not None and not self.pseudo_view_valid[idx]:
            return False
        if (self.pseudo_condition_offline
                and int(self.pseudo_condition_labels[idx]) not in (0, 1)):
            return False
        if self.view_valid is None:
            return True
        return self.view_valid[idx] and self.condition_valid[idx]

    def _build_feasible_view_sets(self):
        P = self.batch_size[0]
        if self.pseudo_condition_enabled:
            # No outfit dimension: an identity qualifies when every camera in S
            # holds at least the two sequences a pseudo-condition pair needs.
            self.id_outfit_views = {}
            id_view_counts = {pid: {view: len(idxs) for view, idxs in views.items()}
                              for pid, views in self.id2view2indices.items()}

            if self.pseudo_condition_offline:
                def qualifies(pid, target):
                    cells = self.id2view2condition2indices[pid]
                    return all(
                        len(cells[view][side]) >= self.select_per_view
                        for view in target for side in (0, 1))
            else:
                min_seqs = self.pseudo_condition_builder.min_sequences_per_view(
                    self.select_per_view)

                def qualifies(pid, target):
                    counts = id_view_counts[pid]
                    if not all(counts.get(view, 0) >= min_seqs for view in target):
                        return False
                    if self.pseudo_condition_matching != 'paired_ratio':
                        return True
                    pools = {view: self.id2view2indices[pid][view] for view in target}
                    return self.pseudo_condition_builder.can_host(
                        target, pools, self.select_per_view)

            pool = id_view_counts
            cameras = sorted({view for views in id_view_counts.values() for view in views},
                             key=str)
        else:
            self.id_outfit_views = {
                pid: {outfit: set(views.keys()) for outfit, views in outfits.items()}
                for pid, outfits in self.id2outfit2view2indices.items()}

            def qualifies(pid, target):
                return sum(1 for views in self.id_outfit_views[pid].values()
                           if target <= views) >= self.n_outfit_blocks

            pool = self.id_outfit_views
            cameras = sorted({view for outfits in self.id_outfit_views.values()
                              for views in outfits.values() for view in views}, key=str)

        if len(cameras) < self.V_shared:
            self.feasible_view_sets = {}
            return

        candidates = list(combinations(cameras, self.V_shared))
        if len(candidates) > self.MAX_VIEW_SET_ENUMERATION:
            # Far more camera labels than CCPG has; sample instead of enumerating.
            candidates = [tuple(sorted(random.sample(cameras, self.V_shared), key=str))
                          for _ in range(self.MAX_VIEW_SET_ENUMERATION)]
            candidates = sorted(set(candidates))

        feasible = {}
        for combo in candidates:
            target = set(combo)
            eligible = [pid for pid in pool if qualifies(pid, target)]
            if len(eligible) >= P:
                feasible[combo] = sorted(eligible)
        self.feasible_view_sets = feasible

        if self.rank == 0:
            _log('[NCBSampler] %d of %d view sets of size %d can host a strict '
                 'NCB batch (P=%d, %d candidate identities, %s conditions)'
                 % (len(feasible), len(candidates), self.V_shared, P, len(pool),
                    'pseudo' if self.pseudo_condition_enabled else 'GT'))

    def _normalize_view_label(self, view):
        if isinstance(view, str) and '_' in view:
            return view.split('_')[0]
        return view

    def _build_outfit_view_index(self):
        self.id2outfit2view2indices = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        if self.pseudo_condition_enabled:
            # The GT outfit field must not influence sampling at all, so the index
            # is keyed by view only and the blocks are cut by appearance later.
            self._build_view_index()
            return

        for idx in range(len(self.dataset)):
            if not self._is_ncb_eligible(idx):
                continue
            seq_info = self.dataset.seqs_info[idx]
            label = seq_info[0]
            outfit = seq_info[1]
            view = seq_info[2]
            view_normalized = self._normalize_view_label(view)
            self.id2outfit2view2indices[label][outfit][view_normalized].append(idx)

    def _log_statistics(self):
        """Log sampler statistics at rank 0"""
        print("[NCBSampler] Statistics:")
        if self.pseudo_condition_enabled:
            if self.pseudo_condition_offline:
                per_id = [
                    sum(1 for sides in views.values()
                        if all(len(sides[side]) >= self.select_per_view
                               for side in (0, 1)))
                    for views in self.id2view2condition2indices.values()]
            else:
                per_id = [sum(1 for idxs in views.values() if len(idxs) >= 2)
                          for views in self.id2view2indices.values()]
            print(f"  Total IDs: {len(self.id2view2indices)}")
            source = 'offline labels' if self.pseudo_condition_offline else 'online features'
            print(f"  Mode: pseudo-condition blocks ({source}), V_shared={self.V_shared}, "
                  f"select_per_view={self.select_per_view}, "
                  f"view_source={self.view_source}")
            if per_id:
                print(f"  Eligible A/B views per ID (min/mean/max): "
                      f"({min(per_id)}/{sum(per_id)/len(per_id):.1f}/{max(per_id)})")
            print(f"  Feasible {self.V_shared}-view sets: {len(self.feasible_view_sets)}")
            return

        total_ids = len(self.id2outfit2view2indices)
        print(f"  Total IDs: {total_ids}")
        print(f"  Mode: n_outfit_blocks={self.n_outfit_blocks}, "
              f"V_shared={self.V_shared}, "
              f"select_per_view={self.select_per_view}, "
              f"per_view_shots(pool)={self.per_view_shots}")

        multi_view_outfits = 0
        outfits_per_id = []
        for id_key, outfit_dict in self.id2outfit2view2indices.items():
            n_valid = 0
            for outfit_key, view_dict in outfit_dict.items():
                if len(view_dict) >= self.min_views_per_outfit:
                    multi_view_outfits += 1
                    n_valid += 1
            outfits_per_id.append(n_valid)

        print(f"  Outfit groups with >={self.min_views_per_outfit} views: {multi_view_outfits}")
        if outfits_per_id:
            ids_enough = sum(1 for n in outfits_per_id if n >= self.n_outfit_blocks)
            print(f"  IDs with >={self.n_outfit_blocks} valid outfits: "
                  f"{ids_enough}/{total_ids}")
            print(f"  Valid outfits/ID (min/mean/max): "
                  f"({min(outfits_per_id)}/"
                  f"{sum(outfits_per_id)/len(outfits_per_id):.1f}/"
                  f"{max(outfits_per_id)})")

    def _select_shared_views(self, pid_outfit_pairs):
        """
        Select V_shared views that maximize coverage across all (pid, outfit) pairs.

        Strategy:
        1. Try intersection (perfect coverage)
        2. Filter by ensure_min_shared_coverage threshold
        3. Select top V_shared by coverage
        """
        available_views_per_pair = []
        for pid, outfit in pid_outfit_pairs:
            view_dict = self.id2outfit2view2indices[pid][outfit]
            available_views_per_pair.append(set(view_dict.keys()))

        if len(available_views_per_pair) > 0:
            intersection_views = set.intersection(*available_views_per_pair)
            if len(intersection_views) >= self.V_shared:
                shared_views_list = sync_random_sample_list(
                    sorted(list(intersection_views)), self.V_shared)
                return shared_views_list, None

        view_coverage = defaultdict(int)
        for view_set in available_views_per_pair:
            for view in view_set:
                view_coverage[view] += 1

        if self.ensure_min_shared_coverage > 0:
            candidate_views = {v: c for v, c in view_coverage.items()
                             if c >= self.ensure_min_shared_coverage}
            if len(candidate_views) >= self.V_shared:
                view_coverage = candidate_views

        sorted_views = sorted(view_coverage.items(),
                            key=lambda x: (-x[1], str(x[0])))
        shared_views = [view for view, _ in sorted_views[:self.V_shared]]

        return shared_views, dict(view_coverage)

    def _sample_from_shared_views(self, pid, outfit, shared_views, block_seed=None):
        view_dict = self.id2outfit2view2indices[pid][outfit]
        sampled_indices = []
        fallback_count = 0
        unique_views_used = set()
        view_selections = {}

        rng = random.Random(block_seed) if (self.deterministic_selection and block_seed is not None) else random
        n_select = self.select_per_view

        for view in shared_views:
            if view in view_dict and len(view_dict[view]) > 0:
                indices_for_view = view_dict[view]

                # Step 1: Build candidate pool (per_view_shots candidates)
                if len(indices_for_view) >= self.per_view_shots:
                    candidates = rng.sample(indices_for_view, self.per_view_shots)
                else:
                    candidates = rng.choices(indices_for_view, k=self.per_view_shots)

                # Step 2: View-stratified sub-selection
                if n_select >= len(candidates):
                    selected = candidates[:n_select]
                else:
                    selected = rng.sample(candidates, n_select)

                sampled_indices.extend(selected)
                unique_views_used.add(view)
                view_selections[view] = {
                    'selected': selected, 'pool': len(indices_for_view), 'fallback': False}
            else:
                fallback_count += n_select
                fb_view = None

                if self.fallback_policy == 'coverage_top':
                    available_views = [v for v in view_dict.keys() if len(view_dict[v]) > 0]
                    if available_views:
                        fallback_view = rng.choice(available_views)
                        fb_view = fallback_view
                        indices_for_view = view_dict[fallback_view]
                        if len(indices_for_view) >= n_select:
                            sampled = rng.sample(indices_for_view, n_select)
                        else:
                            sampled = rng.choices(indices_for_view, k=n_select)
                        sampled_indices.extend(sampled)
                        unique_views_used.add(fallback_view)
                elif self.fallback_policy == 'random_available':
                    all_indices = []
                    for v, idx_list in view_dict.items():
                        all_indices.extend(idx_list)
                    if len(all_indices) >= n_select:
                        sampled = rng.sample(all_indices, n_select)
                    else:
                        sampled = rng.choices(all_indices, k=n_select)
                    sampled_indices.extend(sampled)
                    unique_views_used.update(view_dict.keys())
                    fb_view = 'mixed'

                view_selections[view] = {
                    'selected': sampled if fb_view else [],
                    'pool': 0, 'fallback': True, 'fallback_view': fb_view}

        return sampled_indices, fallback_count, len(unique_views_used), view_selections

    # ---------------- strict_resample_then_pk ----------------

    def _draw_pid_outfit_pairs(self):

        eligible_pids = list(self.id2outfit2view2indices.keys())
        if len(eligible_pids) < self.batch_size[0]:
            return None

        pid_outfit_pairs = []
        for pid in random.sample(eligible_pids, self.batch_size[0]):
            outfit_dict = self.id2outfit2view2indices[pid]
            candidates = [outfit for outfit, view_dict in outfit_dict.items()
                          if len(view_dict) >= self.min_views_per_outfit]
            if len(candidates) < self.n_outfit_blocks:
                return None
            pid_outfit_pairs.extend(
                (pid, outfit)
                for outfit in random.sample(candidates, self.n_outfit_blocks))
        return pid_outfit_pairs

    def _exact_shared_views(self, pid_outfit_pairs):
        """Views present in EVERY (pid, outfit) block. No coverage relaxation."""
        available = [set(self.id2outfit2view2indices[pid][outfit].keys())
                     for pid, outfit in pid_outfit_pairs]
        if not available:
            return set()
        return set.intersection(*available)

    def _identity_shared_views(self, identity_pairs):
        available = [set(self.id2outfit2view2indices[pid][outfit].keys())
                     for pid, outfit in identity_pairs]
        if not available:
            return set()
        return set.intersection(*available)

    def _choose_shared_views_strict(self, intersection):
        """Pick V_shared views from the exact intersection (rank 0 only)."""
        ordered = sorted(intersection, key=str)
        if self.view_selection_mode == 'soft_exposure_regularization':
            combos = list(combinations(ordered, self.V_shared))
            pool = sorted(set(ordered) | set(self.view_exposure), key=str)
            scores = []
            for combo in combos:
                proposed = [self.view_exposure[view] + int(view in combo)
                            for view in pool]
                scores.append(-_coefficient_of_variation(proposed)
                              - 0.25 * _gini(proposed))
            best = max(scores)
            tied = [combo for combo, score in zip(combos, scores)
                    if score >= best - 1e-12]
            return list(random.choice(tied))
        if self.view_selection_mode == 'inverse_exposure':
            weights = [1.0 / (1.0 + self.view_exposure[view]) for view in ordered]
            chosen, pool, pool_w = [], list(ordered), list(weights)
            for _ in range(self.V_shared):
                pick = random.choices(range(len(pool)), weights=pool_w, k=1)[0]
                chosen.append(pool.pop(pick))
                pool_w.pop(pick)
            return chosen
        return random.sample(ordered, self.V_shared)

    def _draw_strict_block(self, pid, outfit, shared_views):
        """Draw select_per_view samples per shared view, never leaving the cell."""
        view_dict = self.id2outfit2view2indices[pid][outfit]
        indices, n_draws, n_with_replacement = [], 0, 0
        for view in shared_views:
            pool = view_dict[view]
            if not pool:
                # Cannot happen: shared_views come from the exact intersection.
                raise RuntimeError(
                    "strict policy hit an empty cell (%s, %s, %s)" % (pid, outfit, view))
            n_draws += self.select_per_view
            if len(pool) >= self.select_per_view:
                indices.extend(random.sample(pool, self.select_per_view))
            else:
                n_with_replacement += self.select_per_view
                indices.extend(random.choices(pool, k=self.select_per_view))
        return indices, n_draws, n_with_replacement

    def _standard_pk_indices(self, pids=None):
        P, K = self.batch_size
        pool_ids = pids if pids else self.dataset.label_set
        if len(pool_ids) < P:
            pool_ids = self.dataset.label_set
        indices = []
        for pid in random.sample(pool_ids, P):
            pool = self.dataset.indices_dict[pid]
            self.identity_exposure[pid] += 1
            if len(pool) >= K:
                indices.extend(random.sample(pool, K))
            else:
                indices.extend(random.choices(pool, k=K))
        return indices

    def _choose_feasible_view_set(self):
        """Pick one feasible camera set, optionally favouring under-exposed ones."""
        combos = sorted(self.feasible_view_sets.keys())
        if self.view_selection_mode == 'soft_exposure_regularization':
            pool = sorted({view for combo in combos for view in combo}, key=str)
            scores = []
            for combo in combos:
                proposed = [self.view_exposure[view] + int(view in combo)
                            for view in pool]
                # Match the frozen search: exposure balance is primary and the
                # number of eligible identities only breaks practically equal
                # choices.
                scores.append(-_coefficient_of_variation(proposed)
                              - 0.25 * _gini(proposed)
                              + 1e-6 * math.log1p(
                                  len(self.feasible_view_sets[combo])))
            best = max(scores)
            tied = [combo for combo, score in zip(combos, scores)
                    if score >= best - 1e-12]
            return random.choice(tied)
        if self.view_selection_mode == 'inverse_exposure':
            weights = [1.0 / (1.0 + sum(self.view_exposure[v] for v in combo))
                       for combo in combos]
            return random.choices(combos, weights=weights, k=1)[0]
        return random.choice(combos)

    def _draw_feasible_view_first(self):
        if not self.feasible_view_sets:
            return None
        shared_views = self._choose_feasible_view_set()
        target = set(shared_views)
        eligible = self.feasible_view_sets[shared_views]

        pid_outfit_pairs = []
        for pid in random.sample(eligible, self.batch_size[0]):
            outfits = [outfit for outfit, views in self.id_outfit_views[pid].items()
                       if target <= views]
            pid_outfit_pairs.extend(
                (pid, outfit)
                for outfit in random.sample(outfits, self.n_outfit_blocks))
        return pid_outfit_pairs, list(shared_views)

    def _draw_strict_ncb(self):
        """One strict NCB attempt under whichever draw strategy is active."""
        if self.partial_enabled:
            drawn = self._draw_feasible_view_first()
            if drawn is None:
                self.n_trials_no_shared_views += 1
            return drawn

        pid_outfit_pairs = self._draw_pid_outfit_pairs()
        if pid_outfit_pairs is None:
            self.n_trials_no_distinct_conditions += 1
            return None
        intersection = self._exact_shared_views(pid_outfit_pairs)
        self.best_intersection_size = max(self.best_intersection_size, len(intersection))
        if len(intersection) < self.V_shared:
            self.n_trials_no_shared_views += 1
            return None
        return pid_outfit_pairs, self._choose_shared_views_strict(intersection)

    def _standard_pk_block(self, pid, size):
        """Draw ``size`` Standard-PK slots for one already selected identity."""
        if size <= 0:
            return []
        pool = self.dataset.indices_dict[pid]
        if len(pool) >= size:
            return random.sample(pool, size)
        return random.choices(pool, k=size)

    def _build_selected_blocks(self, pid_outfit_pairs):
        block_size = self.V_shared * self.select_per_view
        indices, plan = [], []
        n_draws = n_with_replacement = n_pk = 0
        tiers = {'full': 0, 'partial': 0, 'pk': 0}

        for start in range(0, len(pid_outfit_pairs), self.n_outfit_blocks):
            identity_pairs = pid_outfit_pairs[start:start + self.n_outfit_blocks]
            pid = identity_pairs[0][0]
            if len(identity_pairs) != self.n_outfit_blocks or any(
                    other_pid != pid for other_pid, _ in identity_pairs):
                raise RuntimeError(
                    "identity/condition blocks must be contiguous and complete")

            intersection = self._identity_shared_views(identity_pairs)
            self.best_intersection_size = max(
                self.best_intersection_size, len(intersection))
            if len(intersection) >= self.V_shared:
                shared_views = self._choose_shared_views_strict(intersection)
                kind = 'full'
            elif intersection:
                shared_views = sorted(intersection, key=str)
                kind = 'partial'
            else:
                shared_views = []
                kind = 'pk'
            tiers[kind] += 1
            for view in shared_views:
                self.view_exposure[view] += 1

            controlled_blocks = []
            for pair_offset, (_, outfit) in enumerate(identity_pairs):
                block_index = start + pair_offset
                block_seed = None
                if self.deterministic_selection:
                    block_seed = hash(
                        (self.iteration, pid, outfit, block_index)) & 0xFFFFFFFF
                controlled, fallback, _, _ = self._sample_from_shared_views(
                    pid, outfit, shared_views, block_seed=block_seed)
                if fallback:
                    raise RuntimeError(
                        "common-view block unexpectedly required fallback")
                controlled_blocks.append((outfit, controlled))
                n_draws += len(controlled)
                n_with_replacement += len(controlled) - len(set(controlled))

            fill_per_block = block_size - len(controlled_blocks[0][1])
            pk_for_identity = self._standard_pk_block(
                pid, fill_per_block * self.n_outfit_blocks)
            n_pk += len(pk_for_identity)

            for block_offset, (outfit, controlled) in enumerate(controlled_blocks):
                pk_start = block_offset * fill_per_block
                pk = pk_for_identity[pk_start:pk_start + fill_per_block]
                block = controlled + pk
                if len(block) != block_size:
                    raise RuntimeError(
                        "partial controlled block has %d samples, expected %d"
                        % (len(block), block_size))
                indices.extend(block)
                plan.append({
                    'identity': pid,
                    'condition': outfit,
                    'kind': kind,
                    'shared_views': list(shared_views),
                    'controlled_indices': list(controlled),
                    'pk_indices': list(pk),
                    'indices': list(block),
                })

        return indices, plan, n_draws, n_with_replacement, n_pk, tiers

    def _build_partial_controlled_batch(self):
        """Rank-0 block-first NCB construction with per-identity PK fill."""
        P, K = self.batch_size
        meta = {'trials': 0, 'used_pk': 0, 'scheduled_pk': 0,
                'draws': 0, 'draws_repl': 0,
                'id_full': 0, 'id_partial': 0, 'id_pk': 0}

        for trial in range(1, self.max_resample_trials + 1):
            meta['trials'] = trial
            pid_outfit_pairs = self._draw_pid_outfit_pairs()
            if pid_outfit_pairs is None:
                self.n_trials_no_distinct_conditions += 1
                continue

            (indices, self.last_block_plan, meta['draws'], meta['draws_repl'],
             _, tiers) = self._build_selected_blocks(pid_outfit_pairs)
            meta['id_full'] = tiers['full']
            meta['id_partial'] = tiers['partial']
            meta['id_pk'] = tiers['pk']
            # Only a batch in which EVERY identity lost its intersection counts
            # as a whole-batch PK fallback.
            meta['used_pk'] = int(tiers['pk'] == P)
            if meta['used_pk']:
                self.n_trials_no_shared_views += 1
            self.last_shared_views = sorted(
                {view for entry in self.last_block_plan
                 for view in entry['shared_views']}, key=str)
            for pid, _ in pid_outfit_pairs[::self.n_outfit_blocks]:
                self.identity_exposure[pid] += 1
            assert len(indices) == P * K, (
                "partial controlled NCB batch has %d samples, expected %d"
                % (len(indices), P * K))
            return indices, meta

        # This branch means identity/condition blocks themselves could not be
        # selected. There is no controlled block to preserve, so use ordinary PK.
        meta['used_pk'] = 1
        meta['id_pk'] = P
        self.last_block_plan = []
        self.last_shared_views = []
        indices = self._standard_pk_indices()
        assert len(indices) == P * K
        return indices, meta

    def _try_pseudo_condition_batch(self):
        """One attempt at a pseudo-condition NCB batch, or None.

        Layout per identity: for each of the V_shared view groups, S shots of
        pseudo-condition A and S shots of pseudo-condition B, where the A/B split
        comes from GEI appearance only. A and B are always two different
        sequences, so a view never contributes the same sequence to both blocks.
        """
        if not self.feasible_view_sets:
            return None
        shared_views = self._choose_feasible_view_set()
        eligible = list(self.feasible_view_sets[shared_views])
        random.shuffle(eligible)

        block_a, block_b, plan = [], [], []
        for pid in eligible:
            if len(plan) >= self.batch_size[0]:
                break
            if self.pseudo_condition_offline:
                assignment = {}
                valid = True
                for view in shared_views:
                    sides = self.id2view2condition2indices[pid][view]
                    if any(len(sides[side]) < self.select_per_view
                           for side in (0, 1)):
                        valid = False
                        break
                    assignment[view] = (
                        random.sample(sides[0], self.select_per_view),
                        random.sample(sides[1], self.select_per_view))
                if not valid:
                    continue
                for view in shared_views:
                    side_a, side_b = assignment[view]
                    block_a.extend(side_a)
                    block_b.extend(side_b)
                plan.append({
                    'identity': pid,
                    'reference_view': None,
                    'prototypes': None,
                    'shared_views': list(shared_views),
                    'assignment': assignment,
                })
                continue

            pools = {view: self.id2view2indices[pid][view] for view in shared_views}
            built = self.pseudo_condition_builder.build_block(
                shared_views, pools, shots_per_condition=self.select_per_view)
            if built is None:
                continue
            sides = []
            valid = True
            for view in shared_views:
                side_a, side_b = built['assignment'][view]
                if (len(side_a) != self.select_per_view
                        or len(side_b) != self.select_per_view
                        or len(set(side_a + side_b)) != 2 * self.select_per_view):
                    valid = False
                    break
                sides.append((side_a, side_b))
            if not valid:
                continue
            for side_a, side_b in sides:
                block_a.extend(side_a)
                block_b.extend(side_b)
            plan.append({'identity': pid, 'reference_view': built['reference_view'],
                         'prototypes': built['prototypes'],
                         'shared_views': list(shared_views),
                         'assignment': dict(built['assignment'])})

        if len(plan) < self.batch_size[0]:
            return None

        # Exposure is only credited once the whole batch stands, so abandoned
        # attempts do not skew the histograms.
        for entry in plan:
            self.identity_exposure[entry['identity']] += 1
        for view in shared_views:
            self.view_exposure[view] += 1
        self.last_block_plan = plan
        return block_a + block_b

    def _build_strict_batch(self):
        """Rank-0 construction of one strict batch. Returns (indices, meta)."""
        P, K = self.batch_size
        meta = {'trials': 0, 'used_pk': 0, 'scheduled_pk': 0, 'draws': 0, 'draws_repl': 0}

        if self.pseudo_condition_enabled:
            n_cells = P * self.V_shared * self.n_outfit_blocks
            for trial in range(1, self.max_resample_trials + 1):
                meta['trials'] = trial
                indices = self._try_pseudo_condition_batch()
                if indices is None:
                    self.n_trials_no_shared_views += 1
                    continue
                meta['draws'] = n_cells * self.select_per_view
                if self.pseudo_condition_matching in (
                        'joint_k_nearest', 'halfspace_random', 'paired_ratio',
                        'offline_assignment'):
                    meta['draws_repl'] = 0
                else:
                    meta['draws_repl'] = n_cells * self.select_per_view
                assert len(indices) == P * K, (
                    'pseudo-condition NCB batch has %d samples, expected %d'
                    % (len(indices), P * K))
                return indices, meta
            meta['used_pk'] = 1
            self.last_block_plan = []
            indices = self._standard_pk_indices()
            assert len(indices) == P * K
            return indices, meta

        # Proportional schedule: annotated and unannotated identities are exposed
        # in proportion to their share of the training set, so the unannotated
        # half is not starved by NCB batches it can never take part in.
        if self.partial_enabled and random.random() >= self.joint_identity_ratio:
            meta['scheduled_pk'] = 1
            indices = self._standard_pk_indices(self.complement_identities)
            assert len(indices) == P * K
            return indices, meta

        for trial in range(1, self.max_resample_trials + 1):
            meta['trials'] = trial
            drawn = self._draw_strict_ncb()
            if drawn is None:
                continue
            pid_outfit_pairs, shared_views = drawn

            indices = []
            for pid, outfit in pid_outfit_pairs:
                block, draws, repl = self._draw_strict_block(pid, outfit, shared_views)
                indices.extend(block)
                meta['draws'] += draws
                meta['draws_repl'] += repl
            for view in shared_views:
                self.view_exposure[view] += 1
            for pid, _ in pid_outfit_pairs[::self.n_outfit_blocks]:
                self.identity_exposure[pid] += 1
            assert len(indices) == P * K, (
                "strict NCB batch has %d samples, expected %d" % (len(indices), P * K))
            return indices, meta

        meta['used_pk'] = 1
        indices = self._standard_pk_indices()
        assert len(indices) == P * K, (
            "Standard PK fallback produced %d samples, expected %d" % (len(indices), P * K))
        return indices, meta

    def _broadcast_batch(self, indices, meta):
        """Share the rank-0 batch with every rank in a single collective."""
        P, K = self.batch_size
        n = P * K
        fields = ('trials', 'used_pk', 'scheduled_pk', 'draws', 'draws_repl',
                  'id_full', 'id_partial', 'id_pk')
        payload = torch.zeros(n + len(fields), dtype=torch.long)
        if self.rank == 0:
            payload[:n] = torch.tensor(indices, dtype=torch.long)
            for offset, field in enumerate(fields):
                payload[n + offset] = meta.get(field, 0)
        if torch.cuda.is_available():
            payload = payload.cuda()
        torch.distributed.broadcast(payload, src=0)
        payload = payload.tolist()
        return payload[:n], {field: payload[n + offset]
                             for offset, field in enumerate(fields)}

    def _strict_iter(self):
        P, K = self.batch_size
        while True:
            indices, meta = (self._build_strict_batch() if self.rank == 0
                             else (None, None))
            indices, meta = self._broadcast_batch(indices, meta)

            self.n_batches += 1
            self.n_resample_trials += meta['trials']
            self.n_cell_draws += meta['draws']
            self.n_cell_draws_with_replacement += meta['draws_repl']
            if meta['scheduled_pk']:
                self.n_scheduled_pk += 1
            elif meta['used_pk']:
                self.n_standard_pk += 1
            elif meta['trials'] == 1:
                self.n_perfect_intersection += 1
            else:
                self.n_resample_success += 1

            assert len(indices) == P * K, (
                "NCB batch has %d samples, expected %d" % (len(indices), P * K))
            self.last_batch_size = len(indices)

            if self.batch_shuffle:
                indices = sync_random_sample_list(indices, len(indices))

            total_size = int(math.ceil(len(indices) / self.world_size)) * self.world_size
            indices = indices + indices[:(total_size - len(indices))]
            rank_indices = indices[self.rank:total_size:self.world_size]

            if self.log_batch_stats and self.rank == 0 and self.iteration % 200 == 0:
                stats = self.get_stats()
                _log("[NCBSampler iter {}] ncb_batch={:.3f} scheduled_pk={:.3f} "
                     "strict_ncb_failure_fallback={:.3f} total_pk={:.3f} "
                     "perfect_intersection={:.3f} resample_success={:.3f} "
                     "mean_resample_trials={:.2f} within_cell_repl={:.3f} "
                     "actual_batch_size={}".format(
                         self.iteration, stats['ncb_batch_ratio'],
                         stats['scheduled_pk_ratio'],
                         stats['strict_ncb_failure_fallback_ratio'],
                         stats['total_pk_ratio'],
                         stats['perfect_intersection_ratio'],
                         stats['resample_success_ratio'],
                         stats['mean_resample_trials'],
                         stats['within_cell_replacement_ratio'],
                         stats['actual_batch_size']))
                if self.iteration % 1000 == 0:
                    _log("[NCBSampler iter {}] camera_exposure={}".format(
                        self.iteration, stats['camera_exposure']))
                    _log("[NCBSampler iter {}] identity_exposure "
                         "min/mean/max={}/{:.1f}/{} over {} ids".format(
                             self.iteration,
                             min(stats['identity_exposure'].values(), default=0),
                             (sum(stats['identity_exposure'].values())
                              / max(len(stats['identity_exposure']), 1)),
                             max(stats['identity_exposure'].values(), default=0),
                             len(stats['identity_exposure'])))

            self.iteration += 1
            yield rank_indices

    def _partial_controlled_iter(self):
        """DDP iterator for block-first common-view control with PK slot fill."""
        P, K = self.batch_size
        expected = P * K
        while True:
            indices, meta = (self._build_partial_controlled_batch()
                             if self.rank == 0 else (None, None))
            indices, meta = self._broadcast_batch(indices, meta)

            controlled_slots = meta['draws']
            pk_fill_slots = expected - controlled_slots
            self.n_batches += 1
            self.n_resample_trials += meta['trials']
            self.n_cell_draws += controlled_slots
            self.n_cell_draws_with_replacement += meta['draws_repl']
            self.n_partial_controlled_slots += controlled_slots
            self.n_partial_controlled_pk_fill_slots += pk_fill_slots
            if controlled_slots == expected:
                self.n_partial_controlled_full += 1
            elif controlled_slots > 0:
                self.n_partial_controlled_partial += 1
            else:
                self.n_partial_controlled_pk += 1
            self.n_identity_full += meta['id_full']
            self.n_identity_partial += meta['id_partial']
            self.n_identity_pk += meta['id_pk']

            assert len(indices) == expected, (
                "partial controlled batch has %d samples, expected %d"
                % (len(indices), expected))
            self.last_batch_size = len(indices)

            if self.batch_shuffle:
                indices = sync_random_sample_list(indices, len(indices))

            total_size = int(math.ceil(len(indices) / self.world_size)) * self.world_size
            indices = indices + indices[:(total_size - len(indices))]
            rank_indices = indices[self.rank:total_size:self.world_size]

            if self.log_batch_stats and self.rank == 0 and self.iteration % 200 == 0:
                stats = self.get_stats()
                _log("[NCBSampler iter {}] id_full={:.3f} id_partial={:.3f} "
                     "id_pk={:.3f} full={:.3f} partial={:.3f} pk={:.3f} "
                     "controlled_slots={:.3f} pk_fill_slots={:.3f} "
                     "mean_trials={:.2f} actual_batch_size={}".format(
                         self.iteration,
                         stats['full_identity_ratio'],
                         stats['partial_identity_ratio'],
                         stats['pk_identity_ratio'],
                         stats['full_ncb_batch_ratio'],
                         stats['partial_controlled_batch_ratio'],
                         stats['standard_pk_batch_ratio'],
                         stats['controlled_slot_ratio'],
                         stats['pk_fill_slot_ratio'],
                         stats['mean_resample_trials'],
                         stats['actual_batch_size']))

            self.iteration += 1
            yield rank_indices

    def _get_partial_controlled_stats(self):
        n = max(self.n_batches, 1)
        total_slots = max(
            self.n_batches * self.batch_size[0] * self.batch_size[1], 1)
        n_identities = max(self.n_batches * self.batch_size[0], 1)
        return {
            'full_identity_ratio': self.n_identity_full / n_identities,
            'partial_identity_ratio': self.n_identity_partial / n_identities,
            'pk_identity_ratio': self.n_identity_pk / n_identities,
            'full_ncb_batch_ratio': self.n_partial_controlled_full / n,
            'partial_controlled_batch_ratio':
                self.n_partial_controlled_partial / n,
            'standard_pk_batch_ratio': self.n_partial_controlled_pk / n,
            'controlled_slot_ratio':
                self.n_partial_controlled_slots / total_slots,
            'pk_fill_slot_ratio':
                self.n_partial_controlled_pk_fill_slots / total_slots,
            'mean_resample_trials': self.n_resample_trials / n,
            'within_cell_replacement_ratio':
                self.n_cell_draws_with_replacement / max(self.n_cell_draws, 1),
            'camera_exposure': dict(sorted(
                self.view_exposure.items(), key=lambda kv: str(kv[0]))),
            'identity_exposure': dict(self.identity_exposure),
            'actual_batch_size': self.last_batch_size,
            'n_batches': self.n_batches,
            'failed_trials_no_distinct_conditions':
                self.n_trials_no_distinct_conditions,
            'empty_intersection_batches': self.n_trials_no_shared_views,
            'best_intersection_size_seen': self.best_intersection_size,
        }

    def get_stats(self):
        """Sampler health metrics. Exposure counts are tracked on rank 0."""
        if self.partial_controlled_mode:
            return self._get_partial_controlled_stats()
        n = max(self.n_batches, 1)
        n_ncb = self.n_perfect_intersection + self.n_resample_success
        stats = {
            'ncb_batch_ratio': n_ncb / n,
            'scheduled_pk_ratio': self.n_scheduled_pk / n,
            'strict_ncb_failure_fallback_ratio': self.n_standard_pk / n,
            'total_pk_ratio': (self.n_scheduled_pk + self.n_standard_pk) / n,
            'perfect_intersection_ratio': self.n_perfect_intersection / n,
            'resample_success_ratio': self.n_resample_success / n,
            'standard_pk_fallback_ratio': self.n_standard_pk / n,
            # Averaged over batches that actually attempted an NCB draw; scheduled
            # PK batches never enter the retry loop.
            'mean_resample_trials':
                self.n_resample_trials / max(self.n_batches - self.n_scheduled_pk, 1),
            'within_cell_replacement_ratio':
                self.n_cell_draws_with_replacement / max(self.n_cell_draws, 1),
            'camera_exposure': dict(sorted(self.view_exposure.items(), key=lambda kv: str(kv[0]))),
            'pseudo_view_cluster_exposure': dict(self.view_exposure),
            'identity_exposure': dict(self.identity_exposure),
            'actual_batch_size': self.last_batch_size,
            'n_batches': self.n_batches,
            'n_feasible_view_sets': len(self.feasible_view_sets),
            'failed_trials_no_distinct_conditions': self.n_trials_no_distinct_conditions,
            'failed_trials_no_shared_views': self.n_trials_no_shared_views,
            'best_intersection_size_seen': self.best_intersection_size,
        }
        if self.annotation_stats is not None:
            stats['annotation'] = self.annotation_stats
            stats['jointly_labeled_identity_ratio'] = self.joint_identity_ratio
        return stats

    def __iter__(self):
        if self.strict_mode:
            yield from self._strict_iter()
            return
        if self.partial_controlled_mode:
            yield from self._partial_controlled_iter()
            return
        while True:
            sample_indices = []
            batch_fallback_count = 0
            per_pid_unique_views = []

            # 1) Sample P identities
            pid_list = sync_random_sample_list(
                list(self.id2outfit2view2indices.keys()),
                self.batch_size[0]
            )

            # 2) For each ID, sample n_outfit_blocks outfits
            pid_outfit_pairs = []
            outfits_per_pid = []
            for pid in pid_list:
                outfit_dict = self.id2outfit2view2indices[pid]

                candidate_outfits = [
                    outfit_key for outfit_key, view_dict in outfit_dict.items()
                    if len(view_dict) >= self.min_views_per_outfit
                ]

                if not candidate_outfits:
                    candidate_outfits = list(outfit_dict.keys())

                chosen_outfits = sync_random_sample_list(
                    candidate_outfits, self.n_outfit_blocks)

                for outfit in chosen_outfits:
                    pid_outfit_pairs.append((pid, outfit))
                outfits_per_pid.append(len(set(chosen_outfits)))

            # 3) Select shared views for the batch
            shared_views, view_coverage = self._select_shared_views(pid_outfit_pairs)

            # 4) View-stratified sample from shared views for each block
            for block_idx, (pid, outfit) in enumerate(pid_outfit_pairs):
                block_seed = None
                if self.deterministic_selection:
                    block_seed = hash((self.iteration, pid, outfit, block_idx)) & 0xFFFFFFFF

                sampled, fallback, unique_views, _ = self._sample_from_shared_views(
                    pid, outfit, shared_views, block_seed=block_seed
                )
                sample_indices.extend(sampled)
                batch_fallback_count += fallback
                per_pid_unique_views.append(unique_views)

            # Update stats
            self.total_samples += len(sample_indices)
            self.fallback_samples += batch_fallback_count

            fallback_ratio = batch_fallback_count / len(sample_indices) if len(sample_indices) > 0 else 0
            batch_stats = {
                'shared_views': shared_views,
                'view_coverage': view_coverage,
                'fallback_ratio': fallback_ratio,
                'per_pid_unique_views': per_pid_unique_views,
                'outfits_per_pid': outfits_per_pid
            }
            self.batch_stats_history.append(batch_stats)
            if len(self.batch_stats_history) > 10:
                self.batch_stats_history.pop(0)

            should_log = self.log_batch_stats and self.iteration % 200 == 0

            if should_log and self.rank == 0:
                if view_coverage:
                    coverage_values = list(view_coverage.values())
                    coverage_stats = f"({min(coverage_values)}/{sum(coverage_values)/len(coverage_values):.1f}/{max(coverage_values)})"
                else:
                    coverage_stats = "(perfect_intersection)"

                if per_pid_unique_views:
                    views_stats = f"({min(per_pid_unique_views)}/{sum(per_pid_unique_views)/len(per_pid_unique_views):.1f}/{max(per_pid_unique_views)})"
                else:
                    views_stats = "(N/A)"

                if outfits_per_pid:
                    outfit_stats = f"({min(outfits_per_pid)}/{sum(outfits_per_pid)/len(outfits_per_pid):.1f}/{max(outfits_per_pid)})"
                else:
                    outfit_stats = "(N/A)"

                print(f"[NCBSampler iter {self.iteration}] "
                      f"shared_views={shared_views[:3]}{'...' if len(shared_views) > 3 else ''}, "
                      f"coverage(min/mean/max)={coverage_stats}, "
                      f"fallback_ratio={fallback_ratio:.3f}, "
                      f"per_pid_views(min/mean/max)={views_stats}, "
                      f"unique_outfits/pid={outfit_stats}, "
                      f"n_blocks={self.n_outfit_blocks}, sel/view={self.select_per_view}")

            # 5) Shuffle if requested (recommended for preventing BN bias)
            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            # The strict policy guarantees P*K by construction and asserts it;
            # the legacy policies may fall short, so surface it once instead of
            # crashing a run that is already tolerating flexible batch sizes.
            expected = self.batch_size[0] * self.batch_size[1]
            if len(sample_indices) != expected and not getattr(self, '_warned_batch_size', False):
                self._warned_batch_size = True
                if self.rank == 0:
                    print("[NCBSampler] WARNING: batch has %d samples, expected %d "
                          "(fallback_policy=%s)"
                          % (len(sample_indices), expected, self.fallback_policy))

            # 6) DDP splitting
            total_batch_size = len(sample_indices)
            total_size = int(math.ceil(total_batch_size / self.world_size)) * self.world_size
            sample_indices += sample_indices[:(total_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]

            self.iteration += 1
            yield sample_indices

    def __len__(self):
        return len(self.dataset)
