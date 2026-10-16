# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Оптимизированный stream inference с поддержкой одной и нескольких GPU."""

from __future__ import annotations

import json
import logging
import os
import pprint
import os.path as osp
import pickle as pkl
import shutil
import subprocess
import sys
import tempfile
import time
from bisect import bisect_right
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rekognition_online_action_detection.config.defaults import get_cfg
from rekognition_online_action_detection.engines.lstr.lstr_inference import (
    evaluate_stream_results,
)
from rekognition_online_action_detection.evaluation import compute_result
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.models import transformer as tr
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.parser import assert_and_infer_cfg, parse_args

ENV_OPTIM_WORKER = 'TESTRA_OPTIM_WORKER'
ENV_SESSION_LIST = 'TESTRA_SESSION_LIST'
ENV_STREAM_PKL = 'TESTRA_STREAM_PKL'
ENV_SKIP_GLOBAL_EVAL = 'TESTRA_SKIP_GLOBAL_EVAL'
ENV_SAVE_LOGITS = 'TESTRA_SAVE_LOGITS'
ENV_LOGITS_PATH = 'TESTRA_LOGITS_PATH'

MaskCacheKey = Tuple[int, torch.device, torch.dtype]
MaskCache = Dict[MaskCacheKey, torch.Tensor]


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    if seconds < 60:
        return '{:.2f}s'.format(seconds)
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return '{:.0f}m {:.1f}s'.format(minutes, secs)
    hours, minutes = divmod(minutes, 60)
    return '{:.0f}h {:.0f}m {:.1f}s'.format(hours, minutes, secs)


def load_config(force_stream: bool = True):
    """Load config from CLI without initializing CUDA in the launcher."""
    args = parse_args()
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    assert_and_infer_cfg(cfg, args)
    if force_stream:
        cfg.defrost()
        cfg.MODEL.LSTR.INFERENCE_MODE = 'stream'
        cfg.freeze()
    return cfg, args


def get_session_lengths(cfg, sessions: Sequence[str]) -> Dict[str, int]:
    """Return number of frames per session from target npy files."""
    lengths = {}
    for session in sessions:
        target_path = osp.join(
            cfg.DATA.DATA_ROOT,
            cfg.INPUT.TARGET_PERFRAME,
            session + '.npy',
        )
        lengths[session] = int(np.load(target_path, mmap_mode='r').shape[0])
    return lengths


def lpt_partition(
    sessions: Sequence[str],
    lengths: Dict[str, int],
    num_shards: int,
) -> Tuple[List[List[str]], List[int]]:
    """Partition sessions across shards using longest-processing-time-first."""
    ordered = sorted(sessions, key=lambda s: lengths[s], reverse=True)
    shards: List[List[str]] = [[] for _ in range(num_shards)]
    shard_loads = [0 for _ in range(num_shards)]

    for session in ordered:
        shard_idx = min(range(num_shards), key=lambda i: shard_loads[i])
        shards[shard_idx].append(session)
        shard_loads[shard_idx] += lengths[session]

    return shards, shard_loads


def get_free_gpu_bytes(device: torch.device) -> Optional[int]:
    """Return free GPU memory in bytes if the API is available."""
    if device.type != 'cuda':
        return None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return int(free_bytes)
    except (AttributeError, RuntimeError):
        return None


def estimate_feature_bytes(cfg, num_frames: int) -> int:
    """Estimate raw feature tensor size for one video."""
    modality = cfg.INPUT.MODALITY
    bytes_per_frame = 0
    if 'visual' in modality:
        bytes_per_frame += 2048 * 4
    if 'motion' in modality:
        bytes_per_frame += 1024 * 4
    if 'object' in modality:
        bytes_per_frame += 1024 * 4
    return num_frames * bytes_per_frame * 3


def _tensor_from_numpy(
    array: np.ndarray,
    device: torch.device,
    use_gpu: bool,
) -> torch.Tensor:
    """Create a float tensor on GPU or in pinned CPU memory."""
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if use_gpu:
        return tensor.to(device)
    return tensor.pin_memory()


def load_session_tensors(
    cfg,
    session: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, bool]:
    """Load session features once and place them on GPU or pinned CPU.

    Only loads features for modalities present in ``cfg.INPUT.MODALITY``.
    Unused modalities get a stub tensor (same shape as a loaded one) so that
    downstream code can pass them through without special-casing.
    All arrays are trimmed to the shortest common length to avoid IndexError
    when feature files differ by a frame.
    """
    modality = cfg.INPUT.MODALITY
    _use_visual = 'visual' in modality
    _use_motion = 'motion' in modality
    _use_object = 'object' in modality

    target = np.load(
        osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.TARGET_PERFRAME, session + '.npy'),
    )

    loaded_np = {}
    if _use_visual:
        loaded_np['visual'] = np.load(
            osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.VISUAL_FEATURE, session + '.npy'),
            mmap_mode='r',
        )
    if _use_motion:
        loaded_np['motion'] = np.load(
            osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.MOTION_FEATURE, session + '.npy'),
            mmap_mode='r',
        )
    if _use_object:
        loaded_np['object'] = np.load(
            osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.OBJECT_FEATURE, session + '.npy'),
            mmap_mode='r',
        )

    num_frames = min(target.shape[0], *(a.shape[0] for a in loaded_np.values()))
    target = target[:num_frames]
    for key in loaded_np:
        loaded_np[key] = loaded_np[key][:num_frames]

    stub_key = next(iter(loaded_np))
    stub_np = loaded_np[stub_key]

    visual_np = loaded_np.get('visual', stub_np)
    motion_np = loaded_np.get('motion', stub_np)
    object_np = loaded_np.get('object', stub_np)

    use_gpu = device.type == 'cuda'
    if use_gpu:
        free_bytes = get_free_gpu_bytes(device)
        if free_bytes is not None:
            needed = estimate_feature_bytes(cfg, num_frames)
            use_gpu = needed < free_bytes * 0.6

    try:
        visual = _tensor_from_numpy(np.asarray(visual_np), device, use_gpu)
        motion = _tensor_from_numpy(np.asarray(motion_np), device, use_gpu)
        object_inputs = _tensor_from_numpy(np.asarray(object_np), device, use_gpu)
    except RuntimeError:
        use_gpu = False
        visual = _tensor_from_numpy(np.asarray(visual_np), device, use_gpu)
        motion = _tensor_from_numpy(np.asarray(motion_np), device, use_gpu)
        object_inputs = _tensor_from_numpy(np.asarray(object_np), device, use_gpu)
    return visual, motion, object_inputs, target, use_gpu


def gather_window(
    tensor: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    on_gpu: bool,
) -> torch.Tensor:
    """Gather a window and add batch dimension."""
    idx = torch.as_tensor(indices, device=tensor.device if on_gpu else 'cpu')
    window = tensor.index_select(0, idx)
    if not on_gpu:
        window = window.unsqueeze(0).to(device, non_blocking=True)
    else:
        window = window.unsqueeze(0)
    return window


def get_cached_mask(
    mask_cache: MaskCache,
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a cached causal mask for the given size/device/dtype."""
    key = (size, device, dtype)
    mask = mask_cache.get(key)
    if mask is None:
        mask = tr.generate_square_subsequent_mask(size).to(device=device, dtype=dtype)
        mask_cache[key] = mask
    return mask


def make_optim_stream_inference(model, mask_cache: MaskCache):
    """Patch ``stream_inference`` with mask and work-embedding caches."""
    if getattr(model, '_optim_stream_patched', False):
        return

    def stream_inference(
        long_visual_inputs,
        long_motion_inputs,
        long_object_inputs,
        work_visual_inputs,
        work_motion_inputs,
        work_object_inputs,
        memory_key_padding_mask=None,
        cache_num=1,
        cache_id=0,
    ):
        assert model.long_enabled, 'Long-term memory cannot be empty for stream inference'
        assert len(model.enc_modules) > 0, 'LSTR encoder cannot be disabled for stream inference'

        if ((long_visual_inputs is not None) and
                (long_motion_inputs is not None) and
                (long_object_inputs is not None)):
            long_memories = model.feature_head_long(
                long_visual_inputs if not model.include_work2 else torch.cat(
                    (long_visual_inputs, work_visual_inputs), dim=1,
                ),
                long_motion_inputs if not model.include_work2 else torch.cat(
                    (long_motion_inputs, work_motion_inputs), dim=1,
                ),
                long_object_inputs if not model.include_work2 else torch.cat(
                    (long_object_inputs, work_object_inputs), dim=1,
                ),
            ).transpose(0, 1)

            if model.long_memories_cache is None:
                model.long_memories_cache = [long_memories for _ in range(cache_num)]
            else:
                model.long_memories_cache[cache_id] = torch.cat((
                    model.long_memories_cache[cache_id][1:], long_memories,
                ))

            long_memories = model.long_memories_cache[cache_id]
            pos = model.pos_encoding.pe[:model.long_memory_num_samples, :]

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in model.enc_queries
            ]

            if model.include_work2:
                memory_key_padding_mask = F.pad(
                    memory_key_padding_mask,
                    (0, model.work_memory_num_samples),
                    'constant', 0,
                )
            long_memories = model.enc_modules[0].stream_inference(
                enc_queries[0], long_memories, pos,
                memory_key_padding_mask=memory_key_padding_mask,
                cache_num=cache_num, cache_id=cache_id,
            )
            model.compressed_long_memories_cache = long_memories
            for enc_query, enc_module in zip(enc_queries[1:], model.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)
        else:
            long_memories = model.compressed_long_memories_cache

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in model.enc_queries
            ]

            for enc_query, enc_module in zip(enc_queries[1:], model.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)

        if model.long_enabled:
            memory = long_memories

        if model.work_enabled:
            work_slice = getattr(model, '_optim_work_indices', None)
            precomputed = getattr(model, '_optim_work_feats', None)
            if precomputed is not None and work_slice is not None:
                idx = torch.as_tensor(work_slice, device=precomputed.device)
                work_feats = precomputed.index_select(1, idx)
                work_memories = model.pos_encoding(
                    work_feats.transpose(0, 1),
                    padding=model.long_memory_num_samples if model.long_memory_use_pe else 0,
                )
            else:
                work_memories = model.pos_encoding(
                    model.feature_head_work(
                        work_visual_inputs,
                        work_motion_inputs,
                        work_object_inputs,
                    ).transpose(0, 1),
                    padding=model.long_memory_num_samples if model.long_memory_use_pe else 0,
                )

            if model.dec_query is not None:
                anticipate_memories = model.pos_encoding(
                    model.dec_query.weight.unsqueeze(1).repeat(
                        1, work_memories.shape[1], 1,
                    ),
                    padding=(
                        model.long_memory_num_samples + model.work_memory_num_samples
                        if model.long_memory_use_pe else model.work_memory_num_samples
                    ),
                )
                work_memories = torch.cat((work_memories, anticipate_memories))

            mask = get_cached_mask(
                mask_cache,
                work_memories.shape[0],
                work_memories.device,
                work_memories.dtype,
            )

            if model.long_enabled:
                if model.include_work:
                    extended_memory = torch.cat((memory, work_memories), dim=0)
                    extended_mask = F.pad(
                        mask,
                        (memory.shape[0], 0),
                        'constant', 0,
                    )
                    extended_mask = extended_mask.to(extended_memory.device)
                    output = model.dec_modules(
                        work_memories,
                        memory=extended_memory,
                        tgt_mask=mask,
                        memory_mask=extended_mask,
                    )
                else:
                    output = model.dec_modules(
                        work_memories,
                        memory=memory,
                        tgt_mask=mask,
                    )
            else:
                output = model.dec_modules(
                    work_memories,
                    src_mask=mask,
                )

        score = model.classifier(output)
        return score.transpose(0, 1)

    model.stream_inference = stream_inference
    model._optim_stream_patched = True


def precompute_work_features(
    model,
    visual: torch.Tensor,
    motion: torch.Tensor,
    object_inputs: torch.Tensor,
    device: torch.device,
    on_gpu: bool,
) -> Optional[torch.Tensor]:
    """Precompute pointwise work embeddings for the whole video."""
    if model.include_work2:
        return None

    if on_gpu:
        visual_b = visual.unsqueeze(0)
        motion_b = motion.unsqueeze(0)
        object_b = object_inputs.unsqueeze(0)
    else:
        visual_b = visual.unsqueeze(0).to(device, non_blocking=True)
        motion_b = motion.unsqueeze(0).to(device, non_blocking=True)
        object_b = object_inputs.unsqueeze(0).to(device, non_blocking=True)

    return model.feature_head_work(visual_b, motion_b, object_b)


def build_memory_key_padding_mask(
    long_indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Build memory key padding mask on the target device."""
    memory_key_padding_mask = np.zeros(len(long_indices), dtype=np.float32)
    last_zero = bisect_right(long_indices, 0) - 1
    if last_zero > 0:
        memory_key_padding_mask[:last_zero] = float('-inf')
    return torch.as_tensor(memory_key_padding_mask, device=device).unsqueeze(0)


def finalize_detection_scores(
    score_chunks: List[torch.Tensor],
) -> np.ndarray:
    """Convert GPU score chunks to numpy in baseline-compatible order."""
    if not score_chunks:
        return np.empty((0, 0), dtype=np.float32)
    return torch.cat(score_chunks, dim=0).cpu().numpy()


def save_logits_enabled() -> bool:
    """Return True when raw classifier logits should be saved to disk."""
    return os.environ.get(ENV_SAVE_LOGITS, '').lower() in ('1', 'true', 'yes')


# Порядок классов в logits SagVVit (``labels_list.csv``), не совпадает с TeSTra.
SAGVVIT_CLASS_ORDERS: Dict[str, List[str]] = {
    'TVSeries': [
        'Answer phone', 'Clap', 'Close door', 'Dress up', 'Drink', 'Drive car',
        'Eat', 'Fall/trip', 'Fire weapon', 'Get in/out of car', 'Give something',
        'Go down stairway', 'Go up stairway', 'Hang up phone', 'Kiss', 'Open door',
        'Pick something up', 'Point', 'Pour', 'Punch', 'Read', 'Run', 'Sit down',
        'Smoke', 'Stand up', 'Throw something', 'Undress', 'Use computer', 'Wave',
        'Write', 'background',
    ],
}


def normalize_class_name(name: str) -> str:
    """Normalize class label for matching TeSTra and SagVVit names."""
    return name.strip().lower()


def build_sagvvit_column_perm(
    testra_class_names: Sequence[str],
    sagvvit_class_names: Sequence[str],
) -> List[int]:
    """Build column index perm: SagVVit column ``j`` reads TeSTra column ``perm[j]``."""
    testra_by_norm = {
        normalize_class_name(name): idx
        for idx, name in enumerate(testra_class_names)
    }
    perm: List[int] = []
    for sag_name in sagvvit_class_names:
        key = normalize_class_name(sag_name)
        if key not in testra_by_norm:
            raise ValueError(
                'Class {!r} from SagVVit order is missing in TeSTra CLASS_NAMES'.format(
                    sag_name,
                ),
            )
        perm.append(testra_by_norm[key])
    return perm


def reorder_logits_classes(
    logits: np.ndarray,
    perm: Sequence[int],
) -> np.ndarray:
    """Reorder class axis of per-frame logits to SagVVit ``labels_list`` order."""
    if logits.ndim == 2:
        return logits[:, perm]
    if logits.ndim == 3:
        return logits[:, perm, :]
    raise ValueError(
        'Unsupported logits shape for SagVVit export: {}'.format(logits.shape),
    )


def pack_logits_sagvvit(
    pred_logits_all: Dict[str, np.ndarray],
    sessions: Sequence[str],
    testra_class_names: Sequence[str],
    data_name: str,
) -> List[np.ndarray]:
    """Convert session dict with TeSTra class order to SagVVit ``list`` format."""
    sagvvit_order = SAGVVIT_CLASS_ORDERS.get(data_name)
    if sagvvit_order is None:
        raise ValueError(
            'SagVVit logits export is not configured for DATA_NAME={!r}'.format(
                data_name,
            ),
        )
    perm = build_sagvvit_column_perm(testra_class_names, sagvvit_order)
    sessions_ordered = sorted(sessions)
    missing = [session for session in sessions_ordered if session not in pred_logits_all]
    if missing:
        raise ValueError('Missing logits for sessions: {}'.format(missing))
    return [
        reorder_logits_classes(pred_logits_all[session], perm)
        for session in sessions_ordered
    ]


def save_sagvvit_logits_file(
    path: str,
    pred_logits_all: Dict[str, np.ndarray],
    cfg,
) -> None:
    """Save logits as ``list[np.ndarray]`` in SagVVit stream OAD format."""
    logits_list = pack_logits_sagvvit(
        pred_logits_all,
        cfg.DATA.TEST_SESSION_SET,
        cfg.DATA.CLASS_NAMES,
        cfg.DATA.DATA_NAME,
    )
    torch.save(logits_list, path)


def logits_output_path(
    checkpoint_path: str,
    shard_idx: Optional[int] = None,
) -> str:
    """Return output path for SagVVit-compatible ``logits_stream_task-oad.pth``."""
    override = os.environ.get(ENV_LOGITS_PATH)
    if override:
        return override
    base = osp.splitext(checkpoint_path)[0]
    if shard_idx is not None:
        return '{}.logits_stream_task-oad.gpu{}.pth'.format(base, shard_idx)
    return '{}.logits_stream_task-oad.pth'.format(base)


def scores_for_eval(score: torch.Tensor, data_name: str) -> torch.Tensor:
    """Convert raw classifier output to scores used for per-frame metrics."""
    if data_name.startswith('EK'):
        return score
    return score.softmax(dim=-1)


def append_detection_chunks(
    score: torch.Tensor,
    pred_score_chunks: List[torch.Tensor],
    pred_logit_chunks: Optional[List[torch.Tensor]],
    *,
    work_start: int,
    work_end: int,
    target: np.ndarray,
    gt_targets: List,
    data_name: str,
) -> None:
    """Append per-frame detection scores and optional raw logits."""
    score_t = scores_for_eval(score, data_name)
    if work_start == 0:
        gt_targets.extend(list(target[:work_end]))
        pred_score_chunks.append(score_t)
        if pred_logit_chunks is not None:
            pred_logit_chunks.append(score)
    else:
        gt_targets.append(list(target[work_end - 1]))
        pred_score_chunks.append(score_t[-1].unsqueeze(0))
        if pred_logit_chunks is not None:
            pred_logit_chunks.append(score[-1].unsqueeze(0))


def append_anticipation_chunks(
    score: torch.Tensor,
    pred_score_chunks: List[List[torch.Tensor]],
    pred_logit_chunks: Optional[List[List[torch.Tensor]]],
    *,
    work_start: int,
    work_end: int,
    target: np.ndarray,
    gt_targets: List,
    data_name: str,
    anticipation_num_samples: int,
    work_memory_sample_rate: int,
) -> None:
    """Append per-frame anticipation scores and optional raw logits."""
    score_t = scores_for_eval(score, data_name)
    if work_start == 0:
        upsample_score = score_t[:-anticipation_num_samples].repeat(
            work_memory_sample_rate, dim=0,
        )
        upsample_score = upsample_score[work_start:work_end - 1]
        anticipate_score = score_t[-anticipation_num_samples:]
        for t_a in range(anticipation_num_samples):
            combined = torch.cat(
                (upsample_score, anticipate_score[:t_a + 1]),
                dim=0,
            )
            pred_score_chunks[t_a].append(combined)
        if pred_logit_chunks is not None:
            upsample_logits = score[:-anticipation_num_samples].repeat(
                work_memory_sample_rate, dim=0,
            )
            upsample_logits = upsample_logits[work_start:work_end - 1]
            anticipate_logits = score[-anticipation_num_samples:]
            for t_a in range(anticipation_num_samples):
                combined_logits = torch.cat(
                    (upsample_logits, anticipate_logits[:t_a + 1]),
                    dim=0,
                )
                pred_logit_chunks[t_a].append(combined_logits)
        gt_targets.extend(list(target[:work_end - 1]))
    else:
        for t_a in range(anticipation_num_samples):
            pred_score_chunks[t_a].append(
                score_t[t_a - anticipation_num_samples].unsqueeze(0),
            )
            if pred_logit_chunks is not None:
                pred_logit_chunks[t_a].append(
                    score[t_a - anticipation_num_samples].unsqueeze(0),
                )
        gt_targets.append(list(target[work_end - 1]))


def finalize_anticipation_scores(
    score_chunks: List[List[torch.Tensor]],
    anticipation_num_samples: int,
) -> List[np.ndarray]:
    """Convert GPU anticipation score chunks to numpy arrays."""
    finalized = []
    for t_a in range(anticipation_num_samples):
        chunks = score_chunks[t_a]
        if not chunks:
            finalized.append(np.empty((0, 0), dtype=np.float32))
            continue
        stacked = torch.cat(chunks, dim=0)
        trimmed = stacked[:-(1 + t_a)]
        finalized.append(trimmed.cpu().numpy())
    return finalized


def do_lstr_stream_inference_optim(cfg, model, device, logger) -> None:
    """Run optimized per-frame stream inference."""
    model.eval()
    mask_cache: MaskCache = {}
    make_optim_stream_inference(model, mask_cache)

    long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
    long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
    work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
    work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
    anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES

    pred_scores_all = {}
    gt_targets_all = {}
    pred_logits_all = {} if save_logits_enabled() else None
    total_start = time.time()
    session_times: Dict[str, float] = {}

    with torch.inference_mode():
        for session_idx, session in enumerate(cfg.DATA.TEST_SESSION_SET):
            session_start = time.time()
            logger.info(
                'Session [{}/{}] {}: start'.format(
                    session_idx + 1,
                    len(cfg.DATA.TEST_SESSION_SET),
                    session,
                ),
            )
            model.clear_cache()
            model._optim_work_feats = None
            model._optim_work_indices = None

            visual, motion, object_inputs, target, on_gpu = load_session_tensors(
                cfg, session, device,
            )
            precomputed_work = precompute_work_features(
                model, visual, motion, object_inputs, device, on_gpu,
            )
            if precomputed_work is not None:
                model._optim_work_feats = precomputed_work

            if anticipation_num_samples > 0:
                pred_score_chunks: List[List[torch.Tensor]] = [
                    [] for _ in range(anticipation_num_samples)
                ]
                pred_logit_chunks = (
                    [[] for _ in range(anticipation_num_samples)]
                    if pred_logits_all is not None else None
                )
            else:
                pred_score_chunks = []
                pred_logit_chunks = [] if pred_logits_all is not None else None

            gt_targets: List = []
            long_indices_set = None

            start_time = time.time()

            for work_start, work_end in zip(
                range(0, target.shape[0] + 1),
                range(work_memory_length, target.shape[0] + 1),
            ):
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::work_memory_sample_rate]
                model._optim_work_indices = work_indices

                work_visual_inputs = gather_window(
                    visual, work_indices, device, on_gpu,
                )
                work_motion_inputs = gather_window(
                    motion, work_indices, device, on_gpu,
                )
                work_object_inputs = gather_window(
                    object_inputs, work_indices, device, on_gpu,
                )

                long_end = work_start - long_memory_sample_rate
                if long_end < 0:
                    long_indices = [0 for _ in range(long_memory_num_samples)]
                    long_indices_set = [
                        long_indices for _ in range(long_memory_sample_rate)
                    ]
                    long_visual_inputs = gather_window(
                        visual, np.asarray(long_indices), device, on_gpu,
                    )
                    long_motion_inputs = gather_window(
                        motion, np.asarray(long_indices), device, on_gpu,
                    )
                    long_object_inputs = gather_window(
                        object_inputs, np.asarray(long_indices), device, on_gpu,
                    )
                else:
                    long_indices = (
                        long_indices_set[long_end % long_memory_sample_rate][1:]
                        + [long_end]
                    )
                    long_indices_set[long_end % long_memory_sample_rate] = long_indices
                    long_visual_inputs = gather_window(
                        visual, np.asarray([long_end]), device, on_gpu,
                    )
                    long_motion_inputs = gather_window(
                        motion, np.asarray([long_end]), device, on_gpu,
                    )
                    long_object_inputs = gather_window(
                        object_inputs, np.asarray([long_end]), device, on_gpu,
                    )

                memory_key_padding_mask = build_memory_key_padding_mask(
                    long_indices, device,
                )

                score = model.stream_inference(
                    long_visual_inputs,
                    long_motion_inputs,
                    long_object_inputs,
                    work_visual_inputs,
                    work_motion_inputs,
                    work_object_inputs,
                    memory_key_padding_mask,
                    cache_num=long_memory_sample_rate,
                    cache_id=long_end % long_memory_sample_rate,
                )[0]

                if anticipation_num_samples > 0:
                    append_anticipation_chunks(
                        score,
                        pred_score_chunks,
                        pred_logit_chunks,
                        work_start=work_start,
                        work_end=work_end,
                        target=target,
                        gt_targets=gt_targets,
                        data_name=cfg.DATA.DATA_NAME,
                        anticipation_num_samples=anticipation_num_samples,
                        work_memory_sample_rate=work_memory_sample_rate,
                    )
                else:
                    append_detection_chunks(
                        score,
                        pred_score_chunks,
                        pred_logit_chunks,
                        work_start=work_start,
                        work_end=work_end,
                        target=target,
                        gt_targets=gt_targets,
                        data_name=cfg.DATA.DATA_NAME,
                    )

            elapsed = time.time() - start_time
            session_times[session] = time.time() - session_start
            fps = target.shape[0] / elapsed if elapsed > 0 else 0.0
            logger.info(
                'Session {}: inference {:.3f}s ({:.1f} fps, {} frames)'.format(
                    session, elapsed, fps, target.shape[0],
                ),
            )

            if anticipation_num_samples > 0:
                pred_scores = finalize_anticipation_scores(
                    pred_score_chunks,
                    anticipation_num_samples,
                )
                for t_a in range(anticipation_num_samples):
                    result = compute_result['perframe'](
                        cfg,
                        gt_targets,
                        pred_scores[t_a],
                    )
                    sec = (
                        (t_a + 1) / cfg.DATA.FPS
                        * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
                    )
                    logger.info(
                        'mAP of video ({:.2f}s) {}: {:.5f}'.format(
                            sec, session, result['mean_AP'],
                        ),
                    )
                pred_scores_all[session] = np.stack(pred_scores, axis=0).transpose(1, 2, 0)
            else:
                pred_scores = finalize_detection_scores(pred_score_chunks)
                result = compute_result['perframe'](
                    cfg,
                    gt_targets,
                    pred_scores,
                )
                logger.info(
                    'mAP of video {}: {:.5f}'.format(session, result['mean_AP']),
                )
                pred_scores_all[session] = pred_scores

            if pred_logits_all is not None:
                if anticipation_num_samples > 0:
                    pred_logits = finalize_anticipation_scores(
                        pred_logit_chunks,
                        anticipation_num_samples,
                    )
                    pred_logits_all[session] = np.stack(
                        pred_logits, axis=0,
                    ).transpose(1, 2, 0)
                else:
                    pred_logits_all[session] = finalize_detection_scores(
                        pred_logit_chunks,
                    )

            gt_targets_all[session] = np.array(gt_targets)

            model._optim_work_feats = None
            model._optim_work_indices = None

    total_elapsed = time.time() - total_start
    logger.info(
        'Stream inference finished: {} sessions in {} (avg {:.3f}s/video)'.format(
            len(session_times),
            format_duration(total_elapsed),
            total_elapsed / max(len(session_times), 1),
        ),
    )

    stream_pkl = os.environ.get(ENV_STREAM_PKL)
    if stream_pkl:
        with open(stream_pkl, 'wb') as handle:
            pkl.dump({
                'cfg': cfg,
                'perframe_pred_scores': pred_scores_all,
                'perframe_gt_targets': gt_targets_all,
            }, handle)

    if pred_logits_all is not None:
        logits_path = logits_output_path(cfg.MODEL.CHECKPOINT)
        if os.environ.get(ENV_LOGITS_PATH):
            # Промежуточный shard для multi-GPU merge — dict в порядке TeSTra.
            torch.save(pred_logits_all, logits_path)
            logger.info(
                'Saved raw logits shard for {} sessions to {}'.format(
                    len(pred_logits_all), logits_path,
                ),
            )
        else:
            save_sagvvit_logits_file(logits_path, pred_logits_all, cfg)
            logger.info(
                'Saved SagVVit-format logits for {} sessions to {}'.format(
                    len(pred_logits_all), logits_path,
                ),
            )

    if not os.environ.get(ENV_SKIP_GLOBAL_EVAL):
        eval_start = time.time()
        evaluate_stream_results(cfg, pred_scores_all, gt_targets_all, logger)
        logger.info(
            'Global evaluation finished in {}'.format(
                format_duration(time.time() - eval_start),
            ),
        )


def optim_log_path(checkpoint_path: str) -> str:
    """Return the canonical log path for test_net_optim runs."""
    return osp.splitext(checkpoint_path)[0] + '.optim.txt'


def setup_optim_logger(
    cfg,
    *,
    quiet: bool = False,
    write_file: bool = True,
) -> logging.Logger:
    """Create logger for test_net_optim; file output uses {checkpoint}.optim.txt."""
    logger = logging.getLogger('rekognition.test_net_optim')
    level = logging.DEBUG if cfg.VERBOSE else logging.INFO
    logger.setLevel(level)

    if not logger.handlers:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

        if write_file:
            log_file = optim_log_path(cfg.MODEL.CHECKPOINT)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

    if not quiet:
        logger.info(pprint.pformat(cfg))

    return logger


def run_inference(cfg) -> None:
    """Run optimized stream inference in the current process."""
    total_start = time.time()
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    is_worker = bool(os.environ.get(ENV_OPTIM_WORKER))
    logger = setup_optim_logger(
        cfg,
        quiet=is_worker,
        write_file=not is_worker,
    )

    model_start = time.time()
    model = build_model(cfg, device)
    checkpointer.load(model)
    logger.info(
        'Model loaded in {}'.format(format_duration(time.time() - model_start)),
    )

    do_lstr_stream_inference_optim(cfg, model, device, logger)
    logger.info(
        'Total test time: {}'.format(format_duration(time.time() - total_start)),
    )


def run_worker() -> None:
    """Worker entry point: run optimized inference on a shard of sessions."""
    session_list_path = os.environ[ENV_SESSION_LIST]
    with open(session_list_path, 'r', encoding='utf-8') as handle:
        sessions = json.load(handle)

    cfg, _ = load_config(force_stream=True)
    cfg.defrost()
    cfg.DATA.TEST_SESSION_SET = sessions
    cfg.MODEL.LSTR.INFERENCE_MODE = 'stream'
    cfg.freeze()
    run_inference(cfg)


def merge_shard_results(
    shard_pkls: Sequence[str],
    expected_sessions: Sequence[str],
) -> Tuple[dict, dict]:
    """Load and merge per-GPU shard pickle files."""
    pred_scores_all = {}
    gt_targets_all = {}

    for shard_pkl in shard_pkls:
        with open(shard_pkl, 'rb') as handle:
            shard_data = pkl.load(handle)
        pred_scores_all.update(shard_data['perframe_pred_scores'])
        gt_targets_all.update(shard_data['perframe_gt_targets'])

    expected = set(expected_sessions)
    got = set(pred_scores_all.keys())
    if expected != got:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise RuntimeError(
            'Shard merge mismatch: missing {}, extra {}'.format(missing, extra),
        )

    return pred_scores_all, gt_targets_all


def merge_shard_logits(
    shard_paths: Sequence[str],
    expected_sessions: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Load and merge per-GPU shard logits files."""
    logits_all: Dict[str, np.ndarray] = {}

    for shard_path in shard_paths:
        shard_logits = torch.load(shard_path, map_location='cpu')
        logits_all.update(shard_logits)

    expected = set(expected_sessions)
    got = set(logits_all.keys())
    if expected != got:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise RuntimeError(
            'Logits shard merge mismatch: missing {}, extra {}'.format(
                missing, extra,
            ),
        )

    return logits_all


def run_launcher(cfg, args) -> None:
    """Launch one worker per GPU, merge shard outputs, compute global mAP."""
    gpu_ids = [gpu.strip() for gpu in args.gpu.split(',') if gpu.strip()]
    if not gpu_ids:
        raise RuntimeError('At least one GPU id must be provided via --gpu')

    sessions = list(cfg.DATA.TEST_SESSION_SET)
    lengths = get_session_lengths(cfg, sessions)

    if len(gpu_ids) == 1:
        run_inference(cfg)
        return

    launcher_start = time.time()
    shards, shard_loads = lpt_partition(sessions, lengths, len(gpu_ids))
    logger = setup_optim_logger(cfg, quiet=True, write_file=True)

    logger.info('Balanced optimized stream sharding across {} GPUs'.format(len(gpu_ids)))
    for gpu_id, shard_sessions, shard_load in zip(gpu_ids, shards, shard_loads):
        logger.info(
            'GPU {}: {} videos, {} frames'.format(
                gpu_id, len(shard_sessions), shard_load,
            ),
        )

    checkpoint_base = osp.splitext(cfg.MODEL.CHECKPOINT)[0]
    temp_dir = tempfile.mkdtemp(prefix='testra_optim_')
    processes = []
    shard_pkls = []
    shard_logits = []
    save_logits = save_logits_enabled()

    try:
        script_path = osp.abspath(__file__)
        config_path = osp.abspath(args.config_file)

        for shard_idx, (gpu_id, shard_sessions) in enumerate(zip(gpu_ids, shards)):
            session_list_path = osp.join(temp_dir, 'shard_{}.json'.format(shard_idx))
            shard_pkl_path = '{}.stream.optim.gpu{}.pkl'.format(checkpoint_base, shard_idx)
            shard_pkls.append(shard_pkl_path)
            shard_logits_path = logits_output_path(
                cfg.MODEL.CHECKPOINT, shard_idx=shard_idx,
            )
            if save_logits:
                shard_logits.append(shard_logits_path)

            with open(session_list_path, 'w', encoding='utf-8') as handle:
                json.dump(shard_sessions, handle)

            env = os.environ.copy()
            env[ENV_OPTIM_WORKER] = '1'
            env[ENV_SESSION_LIST] = session_list_path
            env[ENV_STREAM_PKL] = shard_pkl_path
            env[ENV_SKIP_GLOBAL_EVAL] = '1'
            if save_logits:
                env[ENV_SAVE_LOGITS] = '1'
                env[ENV_LOGITS_PATH] = shard_logits_path
            env['CUDA_VISIBLE_DEVICES'] = gpu_id

            cmd = [
                sys.executable,
                script_path,
                '--config_file',
                config_path,
                '--gpu',
                '0',
            ]
            if args.opts is not None:
                cmd.extend(args.opts)

            logger.info('Starting worker {} on GPU {}'.format(shard_idx, gpu_id))
            processes.append(subprocess.Popen(cmd, env=env))

        exit_codes = [process.wait() for process in processes]
        failed = [
            (idx, code) for idx, code in enumerate(exit_codes) if code != 0
        ]
        if failed:
            raise RuntimeError('Workers failed with exit codes: {}'.format(failed))

        workers_elapsed = time.time() - launcher_start
        pred_scores_all, gt_targets_all = merge_shard_results(shard_pkls, sessions)
        if save_logits:
            logits_all = merge_shard_logits(shard_logits, sessions)
            merged_logits_path = logits_output_path(cfg.MODEL.CHECKPOINT)
            save_sagvvit_logits_file(merged_logits_path, logits_all, cfg)
            logger.info(
                'Saved merged SagVVit-format logits for {} sessions to {}'.format(
                    len(logits_all), merged_logits_path,
                ),
            )
        merge_eval_start = time.time()
        evaluate_stream_results(cfg, pred_scores_all, gt_targets_all, logger)
        logger.info(
            'Multi-GPU run finished in {} (workers {}, merge+eval {})'.format(
                format_duration(time.time() - launcher_start),
                format_duration(workers_elapsed),
                format_duration(time.time() - merge_eval_start),
            ),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    if os.environ.get(ENV_OPTIM_WORKER):
        run_worker()
        return

    cfg, args = load_config(force_stream=True)
    run_launcher(cfg, args)


if __name__ == '__main__':
    main()
