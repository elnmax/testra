# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU stream inference launcher with length-balanced video sharding."""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import pickle as pkl
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Sequence, Tuple

import numpy as np

from rekognition_online_action_detection.config.defaults import get_cfg
from rekognition_online_action_detection.engines import do_inference
from rekognition_online_action_detection.engines.lstr.lstr_inference import (
    evaluate_stream_results,
)
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.utils.parser import assert_and_infer_cfg, parse_args

ENV_FAST_WORKER = 'TESTRA_FAST_WORKER'
ENV_SESSION_LIST = 'TESTRA_SESSION_LIST'
ENV_STREAM_PKL = 'TESTRA_STREAM_PKL'
ENV_SKIP_GLOBAL_EVAL = 'TESTRA_SKIP_GLOBAL_EVAL'


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


def run_inference(cfg) -> None:
    """Run stream inference on the current process."""
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    logger = setup_logger(cfg, phase='test')
    model = build_model(cfg, device)
    checkpointer.load(model)
    do_inference(cfg, model, device, logger)


def run_worker() -> None:
    """Worker entry point: run stream inference on a shard of sessions."""
    session_list_path = os.environ[ENV_SESSION_LIST]
    with open(session_list_path, 'r', encoding='utf-8') as f:
        sessions = json.load(f)

    cfg, _ = load_config(force_stream=True)
    cfg.defrost()
    cfg.DATA.TEST_SESSION_SET = sessions
    cfg.MODEL.LSTR.INFERENCE_MODE = 'stream'
    cfg.freeze()
    run_inference(cfg)


def merge_shard_results(
    cfg,
    shard_pkls: Sequence[str],
    expected_sessions: Sequence[str],
) -> Tuple[dict, dict]:
    """Load and merge per-GPU shard pickle files."""
    pred_scores_all = {}
    gt_targets_all = {}

    for shard_pkl in shard_pkls:
        with open(shard_pkl, 'rb') as f:
            shard_data = pkl.load(f)
        pred_scores_all.update(shard_data['perframe_pred_scores'])
        gt_targets_all.update(shard_data['perframe_gt_targets'])

    expected = set(expected_sessions)
    got = set(pred_scores_all.keys())
    if expected != got:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise RuntimeError(
            'Shard merge mismatch: missing {}, extra {}'.format(missing, extra)
        )

    return pred_scores_all, gt_targets_all


def setup_launcher_logger(checkpoint_path: str) -> logging.Logger:
    """Create a lightweight logger for the launcher merge/eval stage."""
    logger = logging.getLogger('rekognition.test_net_fast')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        log_file = osp.splitext(checkpoint_path)[0] + '.fast.txt'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    return logger


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

    shards, shard_loads = lpt_partition(sessions, lengths, len(gpu_ids))
    logger = setup_launcher_logger(cfg.MODEL.CHECKPOINT)

    logger.info('Balanced stream sharding across {} GPUs'.format(len(gpu_ids)))
    for shard_idx, (gpu_id, shard_sessions, shard_load) in enumerate(
        zip(gpu_ids, shards, shard_loads),
    ):
        logger.info(
            'GPU {}: {} videos, {} frames'.format(
                gpu_id, len(shard_sessions), shard_load,
            )
        )

    checkpoint_base = osp.splitext(cfg.MODEL.CHECKPOINT)[0]
    temp_dir = tempfile.mkdtemp(prefix='testra_fast_')
    processes = []
    shard_pkls = []

    try:
        script_path = osp.abspath(__file__)
        config_path = osp.abspath(args.config_file)

        for shard_idx, (gpu_id, shard_sessions) in enumerate(zip(gpu_ids, shards)):
            session_list_path = osp.join(temp_dir, 'shard_{}.json'.format(shard_idx))
            shard_pkl_path = '{}.stream.gpu{}.pkl'.format(checkpoint_base, shard_idx)
            shard_pkls.append(shard_pkl_path)

            with open(session_list_path, 'w', encoding='utf-8') as f:
                json.dump(shard_sessions, f)

            env = os.environ.copy()
            env[ENV_FAST_WORKER] = '1'
            env[ENV_SESSION_LIST] = session_list_path
            env[ENV_STREAM_PKL] = shard_pkl_path
            env[ENV_SKIP_GLOBAL_EVAL] = '1'
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

        pred_scores_all, gt_targets_all = merge_shard_results(
            cfg, shard_pkls, sessions,
        )
        evaluate_stream_results(cfg, pred_scores_all, gt_targets_all, logger)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    if os.environ.get(ENV_FAST_WORKER):
        run_worker()
        return

    cfg, args = load_config(force_stream=True)
    run_launcher(cfg, args)


if __name__ == '__main__':
    main()
