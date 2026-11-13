import glob
import hashlib
import os
import json
import random
import time
from pathlib import Path
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_ogbench_env_and_datasets
from utils.datasets import Dataset, CGCDataset, make_task_dataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'humanoidmaze-giant-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_string('ogbench_dataset_path', None, 'Fixed OGBench training NPZ (optional).')
flags.DEFINE_enum('task_mode', 'goal', ['goal', 'single'], 'Goal-conditioned or fixed-task singletask training.')
flags.DEFINE_enum('critic_mode', 'td', ['td', 'graph'], 'TD bootstrap or fixed graph targets.')
flags.DEFINE_string('graph_targets', None, 'GraphCritic targets.npz/JSON pair for graph critic training.')
flags.DEFINE_string('graph_dataset_id', None, 'Expected dataset ID recorded in graph target metadata.')
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval.')
flags.DEFINE_integer('num_datasets', None, 'Number of datasets to use.')
flags.DEFINE_integer('dataset_size', None, 'Size of the dataset.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 250000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/dqc.py', lock_config=False)

flags.DEFINE_string('tags', 'Default', 'Wandb tag.')

# overriding evaluation parameters
flags.DEFINE_string('action_chunk_eval_sizes', '0', 'separated by commas. 0 is the default. ' \
    'For example: "0,1,5" evaluates with the full policy chunk size first, and then with only ' \
    'the first action in the chunk, and finally with the first 5 actions in the chunk.')
flags.DEFINE_string('best_of_n_eval_values', '0',   'separated by commas. 0 is the default. ' \
    'For example: "0,8,128" evaluates with the agent.best_of_n first, and then with 8 action samples, ' \
    'and finally with 128 action samples.')


def _expected_dataset_id(env_name, dataset_path):
    """Recreate the root graph builder's immutable OGBench dataset identifier."""
    parts = env_name.split('-')
    if 'singletask' not in parts:
        raise ValueError('GraphCritic DQC targets require an OGBench singletask environment')
    pos = parts.index('singletask')
    file_stem = '-'.join(parts[:pos] + parts[-1:])
    selected = Path(dataset_path).expanduser().resolve() if dataset_path else (
        Path('~/.ogbench/data').expanduser() / f'{file_stem}.npz'
    )
    if not selected.is_file():
        raise FileNotFoundError(f'cannot identify the active OGBench dataset: {selected}')
    digest = hashlib.sha256()
    with selected.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    sha256 = digest.hexdigest()
    metadata_path = selected.with_suffix('.json')
    tag = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        declared_hash = metadata.get('dataset_sha256')
        if declared_hash is not None and declared_hash != sha256:
            raise ValueError('dataset archive SHA256 does not match adjacent metadata')
        tag = metadata.get('dataset_tag')
    tag_part = f':{tag}' if tag else ''
    return f'ogbench:{env_name}{tag_part}:{sha256}'


def _chunk_valid(terminals, horizon):
    terminals = np.asarray(terminals, dtype=bool)
    valid = np.zeros(len(terminals), dtype=bool)
    n_starts = max(0, len(terminals) - int(horizon) + 1)
    if n_starts:
        prefix = np.r_[0, np.cumsum(terminals, dtype=np.int64)]
        starts = np.arange(n_starts)
        valid[:n_starts] = (prefix[starts + horizon - 1] - prefix[starts]) == 0
    return valid


def _attach_graph_targets(train_data, target_path, expected_dataset_id, env_name, horizon, gamma,
                          dataset_path=None):
    if target_path is None:
        raise ValueError('--critic_mode=graph requires --graph_targets')
    if expected_dataset_id is None:
        raise ValueError('--graph_dataset_id is required with --graph_targets')
    if expected_dataset_id != _expected_dataset_id(env_name, dataset_path):
        raise ValueError('--graph_dataset_id does not match the active OGBench dataset archive')
    from graphcritic.io import load_targets

    targets = load_targets(
        target_path, expected_dataset_id=expected_dataset_id, require_loss_mask=True
    )
    if targets.dataset_id != expected_dataset_id or targets.unit != 'steps':
        raise ValueError('graph target dataset ID or unit is invalid')
    if targets.kind != 'gamma_pow':
        raise ValueError("DQC graph mode requires targets with kind='gamma_pow'")
    if targets.h != horizon:
        raise ValueError('graph target horizon does not match --agent.backup_horizon')
    if targets.discount_gamma is None or not np.isclose(float(targets.discount_gamma), gamma):
        raise ValueError('graph target gamma does not match --agent.discount')

    transition_rows = np.flatnonzero(np.asarray(train_data['valids']) > 0)
    if targets.y.shape != (len(transition_rows),):
        raise ValueError('graph target rows do not align with the compact dataset transitions')
    transition_terminals = np.asarray(train_data['terminals'])[transition_rows]
    expected_valid = _chunk_valid(transition_terminals, horizon)
    if not np.array_equal(targets.valid, expected_valid):
        raise ValueError('graph target chunk validity does not match DQC transition boundaries')

    n_states = len(train_data['observations'])
    for key, values, dtype in (
        ('graph_target', targets.y, np.float32),
        ('graph_target_valid', targets.valid, np.bool_),
        ('graph_target_loss_mask', targets.loss_mask, np.bool_),
    ):
        aligned = np.zeros(n_states, dtype=dtype)
        aligned[transition_rows] = np.asarray(values, dtype=dtype)
        train_data[key] = aligned
    return targets

def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='dqc', group=FLAGS.run_group, name=exp_name, tags=FLAGS.tags.split(","), mode="offline")

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    if FLAGS.ogbench_dataset_path is not None and FLAGS.dataset_dir is not None:
        raise ValueError('choose either --ogbench_dataset_path or --dataset_dir')
    if FLAGS.ogbench_dataset_path is not None:
        datasets = [FLAGS.ogbench_dataset_path]
    elif FLAGS.dataset_dir is None:
        datasets = [None]
    elif FLAGS.dataset_dir.endswith('npz'):
        # Dataset file.
        datasets = [FLAGS.dataset_dir]
    else:
        # Dataset directory.
        datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
    
    if FLAGS.num_datasets is not None:
        datasets = datasets[:FLAGS.num_datasets]
    
    dataset_idx = 0
    env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
        FLAGS.env_name, dataset_path=datasets[0], compact_dataset=True, add_info=True
    )
    eval_env = make_ogbench_env_and_datasets(
        FLAGS.env_name, dataset_path=datasets[0], compact_dataset=True, env_only=True
    )
    if FLAGS.dataset_size is not None:
        train_dataset = {k: v[:FLAGS.dataset_size] for k, v in train_dataset.items()}

    if FLAGS.task_mode == 'single' and 'singletask' not in FLAGS.env_name:
        raise ValueError('--task_mode=single requires an OGBench singletask environment')
    if FLAGS.critic_mode == 'graph':
        if FLAGS.task_mode != 'single':
            raise ValueError('--critic_mode=graph currently requires --task_mode=single')
        if len(datasets) != 1:
            raise ValueError('graph-target DQC requires one fixed OGBench dataset file')
        targets = _attach_graph_targets(
            train_dataset,
            FLAGS.graph_targets,
            FLAGS.graph_dataset_id,
            FLAGS.env_name,
            int(config['backup_horizon']),
            float(config['discount']),
            dataset_path=datasets[0],
        )
        print({
            'graph_target_kind': targets.kind,
            'graph_target_h': targets.h,
            'graph_target_gamma': targets.discount_gamma,
            'graph_target_dataset_id': targets.dataset_id,
            'graph_target_valid_fraction': float(np.mean(targets.valid)),
            'graph_target_loss_mask_fraction': float(np.mean(targets.loss_mask)),
        }, flush=True)
    elif FLAGS.graph_targets is not None or FLAGS.graph_dataset_id is not None:
        raise ValueError('--graph_targets/--graph_dataset_id require --critic_mode=graph')

    config.critic_mode = FLAGS.critic_mode
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    print(train_dataset.keys())
    print(train_dataset.size)

    env.reset()

    # Clip dataset actions.
    eps = 1e-5
    train_dataset = train_dataset.copy(add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + eps, 1 - eps)))
    if val_dataset is not None:
        val_dataset = val_dataset.copy(add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + eps, 1 - eps)))
    
    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset_class_dict = {'CGCDataset': CGCDataset}
    example_transition = {k: v[0] for k, v in train_dataset.items()}
    example_transition.pop("oracle_reps", None)
    print(example_transition.keys())

    train_dataset = make_task_dataset(
        FLAGS.task_mode, Dataset.create(**train_dataset), config,
        goal_dataset_class=dataset_class_dict[config['dataset_class']],
    )

    example_batch = train_dataset.sample(1)


    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    metric_rng, _ = jax.random.split(jax.random.PRNGKey(FLAGS.seed))
    action_dim = example_batch["actions"].shape[-1]
    for i in tqdm.tqdm(range(0, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)
        
        # Log metrics.
        if i % FLAGS.log_interval == 0:
            if hasattr(agent, 'compute_metrics'):
                metric_rng, key = jax.random.split(metric_rng)
                add_metrics = agent.compute_metrics(batch, rng=key)
                update_info.update({f'metrics/{k}': v for k, v in add_metrics.items()})
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0:
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = len(task_infos)
            eval_tasks = (
                [(None, FLAGS.env_name)] if FLAGS.task_mode == 'single'
                else [(task_id, task_infos[task_id - 1]['task_name'])
                      for task_id in range(1, num_tasks + 1)]
            )
            
            action_chunk_eval_sizes = list(map(int, FLAGS.action_chunk_eval_sizes.split(",")))
            
            eval_metrics = {}
            for action_chunk_eval_size in action_chunk_eval_sizes:
                renders = []
                if action_chunk_eval_size == 0:
                    action_chunk_eval_size = None
                    suffix = ""
                else:
                    if action_chunk_eval_size >= config["policy_chunk_size"]: 
                        print("skipping", action_chunk_eval_size, f"because horizon_length is too short")
                        continue
                    suffix = f"-ac-{action_chunk_eval_size}"
                
                if action_chunk_eval_size is not None:
                    print(f"evaluating with action chunk size of {action_chunk_eval_size}")

                for eval_bfn in list(map(int, FLAGS.best_of_n_eval_values.split(","))):
                    overall_metrics = defaultdict(list)
                    for task_id, task_name in tqdm.tqdm(eval_tasks):

                        eval_info, _, cur_renders = evaluate(
                            agent=agent,
                            agent_name=config['agent_name'],
                            env=eval_env,
                            goal_conditioned=FLAGS.task_mode == 'goal',
                            task_id=task_id,
                            num_eval_episodes=FLAGS.eval_episodes,
                            num_video_episodes=FLAGS.video_episodes,
                            video_frame_skip=FLAGS.video_frame_skip,
                            action_dim=action_dim,
                            action_chunk_eval_size=action_chunk_eval_size,
                            best_of_n_override=eval_bfn if eval_bfn != 0 else None,
                            report_wrong_buttons=FLAGS.task_mode == 'single',
                        )
                        renders.extend(cur_renders)
                        metric_names = (
                            ['success', 'episode.final_wrong_buttons']
                            if FLAGS.task_mode == 'single' else ['success']
                        )

                        if eval_bfn == 0:
                            eval_metrics.update(
                                {f'evaluation{suffix}/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                            )
                        else:
                            eval_metrics.update(
                                {f'evaluation{suffix}-bfn{eval_bfn}/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                            )
                        for k, v in eval_info.items():
                            if k in metric_names:
                                overall_metrics[k].append(v)
                    
                    if eval_bfn == 0:
                        for k, v in overall_metrics.items():
                            eval_metrics[f'evaluation{suffix}/overall_{k}'] = np.mean(v)
                    else:
                        for k, v in overall_metrics.items():
                            eval_metrics[f'evaluation{suffix}-bfn{eval_bfn}/overall_{k}'] = np.mean(v)

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

        if FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            if len(datasets) > 1:
                print(f'Using new dataset: {datasets[dataset_idx]}', flush=True)
                dataset_idx = (dataset_idx + 1) % len(datasets)
                train_dataset, val_dataset = make_ogbench_env_and_datasets(
                    FLAGS.env_name, dataset_path=datasets[dataset_idx], 
                    compact_dataset=True, dataset_only=True, cur_env=env)
                train_dataset = make_task_dataset(
                    FLAGS.task_mode, Dataset.create(**train_dataset), config,
                    goal_dataset_class=dataset_class_dict[config['dataset_class']],
                )

    train_logger.close()
    eval_logger.close()

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        # Offline W&B runs have no URL. Training artifacts are already saved;
        # keep shutdown successful and write an empty token in that case.
        f.write(run.url or '')

if __name__ == '__main__':
    app.run(main)
