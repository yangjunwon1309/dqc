"""Compare saved DQC action critics against fixed h=5 graph targets.

This diagnostic only loads datasets/checkpoints and evaluates critic outputs. It
does not step an environment or update model parameters.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scipy.stats import spearmanr

from agents.dqc import DQCAgent
from main import _attach_graph_targets, _expected_dataset_id
from envs.env_utils import make_ogbench_env_and_datasets
from utils.datasets import CSingleDataset, Dataset
from utils.flax_utils import restore_agent


def _dataset_path(flags):
    path = flags.get('ogbench_dataset_path') or flags.get('dataset_dir')
    if path is None or str(path).endswith('.npz'):
        return path
    candidates = sorted(Path(path).glob('*.npz'))
    candidates = [p for p in candidates if '-val.npz' not in p.name]
    if len(candidates) != 1:
        raise ValueError(f'expected one active training archive in {path}, found {len(candidates)}')
    return str(candidates[0])


def _resolve_run_dir(path):
    path = Path(path)
    if (path / 'flags.json').is_file():
        return path
    candidates = [p.parent for p in path.rglob('flags.json') if (p.parent / 'params_300000.pkl').is_file()]
    if len(candidates) != 1:
        raise ValueError(f'expected one completed 300k checkpoint under {path}, found {len(candidates)}')
    return candidates[0]


def _load_checkpoint(run_dir, target_path):
    run_dir = _resolve_run_dir(run_dir)
    flags = json.loads((run_dir / 'flags.json').read_text())
    env_name = flags['env_name']
    dataset_path = _dataset_path(flags)
    expected_id = _expected_dataset_id(env_name, dataset_path)

    env, raw_data, _ = make_ogbench_env_and_datasets(
        env_name, dataset_path=dataset_path, compact_dataset=True, add_info=True,
    )
    agent_config = ml_collections.ConfigDict(flags['agent'])
    agent_config.critic_mode = flags.get('critic_mode', 'td')
    agent_config.graph_target_level = flags['agent'].get('graph_target_level', 'chunk')
    attach_h = int(agent_config['backup_horizon'])
    attach_gamma = float(agent_config['discount'])
    attached = _attach_graph_targets(
        raw_data, flags['graph_targets'], expected_id, env_name,
        attach_h, attach_gamma, dataset_path=dataset_path,
    )
    if attached.h != attach_h:
        raise ValueError(f'{run_dir}: checkpoint graph target horizon mismatch')
    env.close()

    data = Dataset.create(**raw_data)
    data = data.copy(add_or_replace={
        'actions': np.clip(data['actions'], -1 + 1e-5, 1 - 1e-5),
    })
    dataset_config = {
        'backup_horizon': int(agent_config['backup_horizon']),
        'discount': float(agent_config['discount']),
        'critic_mode': flags.get('critic_mode', 'td'),
    }
    dataset = CSingleDataset(data, dataset_config)
    sample = dataset.sample(1, idxs=np.asarray([int(dataset.dataset.valid_idxs[0])]))
    agent = DQCAgent.create(int(flags.get('seed', 0)), sample, agent_config)
    agent = restore_agent(agent, str(run_dir), 300000)
    return agent, data, expected_id


def diagnose(run_dir, target_path, n_pairs=10000, seed=0, batch_size=1024):
    run_dir = _resolve_run_dir(run_dir)
    agent, data, expected_id = _load_checkpoint(run_dir, target_path)
    from graphcritic.io import load_targets

    targets = load_targets(target_path, expected_dataset_id=expected_id, require_loss_mask=True)
    if targets.h != 5 or targets.kind != 'gamma_pow':
        raise ValueError('diagnostic target must be gamma_pow with h=5')

    transition_rows = np.flatnonzero(np.asarray(data['valids']) > 0)
    aligned_mask = np.zeros(len(data['observations']), dtype=bool)
    aligned_mask[transition_rows] = np.asarray(targets.valid) & np.asarray(targets.loss_mask)
    rows = np.flatnonzero(aligned_mask)
    if len(rows) < n_pairs:
        raise ValueError(f'only {len(rows)} eligible h=5 rows; need {n_pairs}')
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(rows, size=n_pairs, replace=False))

    y_aligned = np.zeros(len(data['observations']), dtype=np.float32)
    y_aligned[transition_rows] = np.asarray(targets.y, dtype=np.float32)
    action_offsets = np.arange(5, dtype=np.int64)
    action_chunks = np.asarray(data['actions'])[rows[:, None] + action_offsets]
    action_chunks = action_chunks.reshape(n_pairs, -1)
    observations = jax.tree_util.tree_map(lambda arr: arr[rows], data['observations'])

    pred_parts = []
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        obs = jax.tree_util.tree_map(lambda arr: jnp.asarray(arr[start:end]), observations)
        actions = jnp.asarray(action_chunks[start:end], dtype=jnp.float32)
        logits = agent.network.select('action_critic')(obs, goals=None, actions=actions)
        if agent.config['q_agg'] == 'mean':
            logits = logits.mean(axis=0)
        elif agent.config['q_agg'] == 'min':
            logits = logits.min(axis=0)
        else:
            raise ValueError(f"unsupported q_agg={agent.config['q_agg']}")
        values = jax.nn.sigmoid(logits)
        pred_parts.append(np.asarray(values, dtype=np.float64))

    prediction = np.concatenate(pred_parts)
    target = y_aligned[rows].astype(np.float64)
    corr = spearmanr(prediction, target)
    return {
        'run_dir': str(Path(run_dir).resolve()),
        'checkpoint': str((Path(run_dir) / 'params_300000.pkl').resolve()),
        'target_path': str(Path(target_path).resolve()),
        'target_kind': targets.kind,
        'target_h': int(targets.h),
        'target_gamma': targets.discount_gamma,
        'q_agg': agent.config['q_agg'],
        'num_qs': int(agent.config['num_qs']),
        'n_pairs': int(n_pairs),
        'seed': int(seed),
        'spearman': float(corr.statistic),
        'mae': float(np.mean(np.abs(prediction - target))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-h5', required=True)
    parser.add_argument('--r2-run', required=True)
    parser.add_argument('--r5-run', required=True)
    parser.add_argument('--r6-run')
    parser.add_argument('--r6-target-h5')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-pairs', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()

    if args.r6_target_h5 and not args.r6_run:
        parser.error('--r6-target-h5 requires --r6-run')
    result = {
        'metric': 'Spearman and MAE of sigmoid(q_agg(action_critic_logits)) against h=5 graph y',
        'runs': [
            diagnose(args.r2_run, args.target_h5, args.n_pairs, args.seed, args.batch_size),
            diagnose(args.r5_run, args.target_h5, args.n_pairs, args.seed, args.batch_size),
        ],
    }
    if args.r6_run:
        result['runs'].append(diagnose(
            args.r6_run, args.r6_target_h5 or args.target_h5,
            args.n_pairs, args.seed, args.batch_size,
        ))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
