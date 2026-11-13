import copy

import jax
import numpy as np
import pytest

from agents.dqc import DQCAgent
from main import _attach_graph_targets, _expected_dataset_id
from utils.datasets import CSingleDataset, CGCDataset, Dataset, make_task_dataset


def _base_fields(n=24):
    terminals = np.zeros(n, dtype=np.float32)
    terminals[-1] = 1.0
    return {
        'observations': np.arange(n * 4, dtype=np.float32).reshape(n, 4) / 10.0,
        'actions': np.tanh(np.arange(n * 2, dtype=np.float32).reshape(n, 2) / 7.0),
        'terminals': terminals,
        'valids': 1.0 - terminals,
    }


def _dataset_config(**overrides):
    config = {
        'backup_horizon': 3,
        'discount': 0.9,
        'value_p_curgoal': 0.2,
        'value_p_trajgoal': 0.5,
        'value_p_randomgoal': 0.3,
        'value_geom_sample': False,
        'critic_mode': 'td',
    }
    config.update(overrides)
    return config


def test_task_mode_goal_reuses_original_cgc_batch_under_same_seed():
    fields = _base_fields()
    original = CGCDataset(Dataset.create(**copy.deepcopy(fields)), _dataset_config())
    selected = make_task_dataset(
        'goal', Dataset.create(**copy.deepcopy(fields)), _dataset_config(),
    )
    np.random.seed(713)
    expected = original.sample(6)
    np.random.seed(713)
    actual = selected.sample(6)
    assert type(selected) is CGCDataset
    assert expected.keys() == actual.keys()
    for key in expected:
        if expected[key] is None:
            assert actual[key] is None
        else:
            np.testing.assert_array_equal(np.asarray(actual[key]), np.asarray(expected[key]))


def _single_fields(n_transitions=12, graph=False):
    n_rows = n_transitions + 1  # compact OGBench layout includes terminal state
    terminals = np.zeros(n_rows, dtype=np.float32)
    terminals[-2:] = 1.0  # terminal transition row and terminal-only state row
    valids = np.ones(n_rows, dtype=np.float32)
    valids[-1] = 0.0
    masks = np.ones(n_rows, dtype=np.float32)
    masks[2] = 0.0  # the state at row two is the fixed-task goal
    fields = {
        'observations': np.arange(n_rows * 4, dtype=np.float32).reshape(n_rows, 4) / 10.0,
        'actions': np.tanh(np.arange(n_rows * 2, dtype=np.float32).reshape(n_rows, 2) / 7.0),
        'terminals': terminals,
        'valids': valids,
        'masks': masks,
        'button_states': np.zeros((n_rows, 9), dtype=np.int64),
    }
    if graph:
        graph_valid = np.zeros(n_transitions, dtype=bool)
        graph_valid[:n_transitions - 3 + 1] = True
        for key, array in (
            ('graph_target', np.full(n_transitions, 0.6, dtype=np.float32)),
            ('graph_target_valid', graph_valid),
            ('graph_target_loss_mask', graph_valid.copy()),
        ):
            aligned = np.zeros(n_rows, dtype=array.dtype)
            aligned[:n_transitions] = array
            fields[key] = aligned
    return fields


def test_single_task_chunk_labels_use_first_success_step_and_do_not_cross_episodes():
    data = Dataset.create(**_single_fields())
    dataset = make_task_dataset('single', data, _dataset_config())
    assert isinstance(dataset, CSingleDataset)
    batch = dataset.sample(2, idxs=np.asarray([0, 3], dtype=np.int64))
    np.testing.assert_allclose(batch['high_value_rewards'], [0.9 ** 2, 0.0], atol=1e-7)
    np.testing.assert_array_equal(batch['high_value_masks'], [0.0, 1.0])
    np.testing.assert_array_equal(batch['high_value_backup_horizon'], [2, 3])
    np.testing.assert_array_equal(batch['high_value_goals'], None)
    assert batch['high_value_action_chunks'].shape == (2, 6)
    assert batch['high_value_next_observations'].shape == (2, 4)


def _changed(left, right):
    return any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    )


def test_graph_update_trains_chunk_and_partial_critics_without_value_or_target_update():
    horizon = 3
    config = _dataset_config(critic_mode='graph')
    data = Dataset.create(**_single_fields(graph=True))
    dataset = make_task_dataset('single', data, config)
    batch = dataset.sample(4)
    agent_config = {
        'lr': 1e-3,
        'value_hidden_dims': (16,),
        'actor_hidden_dims': (16,),
        'layer_norm': False,
        'actor_layer_norm': False,
        'num_qs': 2,
        'use_chunk_critic': True,
        'backup_horizon': horizon,
        'policy_chunk_size': 1,
        'distill_method': 'expectile',
        'kappa_d': 0.5,
        'implicit_backup_type': 'quantile',
        'kappa_b': 0.9,
        'discount': 0.9,
        'tau': 0.005,
        'q_agg': 'mean',
        'flow_steps': 2,
        'best_of_n': 2,
        'critic_mode': 'graph',
    }
    agent = DQCAgent.create(0, batch, agent_config)
    before = agent.network.params
    updated, info = agent.update(batch)
    after = updated.network.params

    assert np.isfinite(float(info['chunk_critic/critic_loss']))
    assert np.isfinite(float(info['action_critic/critic_loss']))
    assert _changed(before['modules_chunk_critic'], after['modules_chunk_critic'])
    assert _changed(before['modules_action_critic'], after['modules_action_critic'])
    assert not _changed(before['modules_value'], after['modules_value'])
    assert not _changed(
        before['modules_target_action_critic'], after['modules_target_action_critic']
    )


def test_graph_target_loader_checks_dataset_rows_horizon_and_validity(tmp_path):
    from graphcritic.io import save_targets
    from graphcritic.types import TargetSet

    env_name = 'puzzle-3x3-play-singletask-task3-v0'
    dataset_path = tmp_path / 'puzzle-3x3-play-v0.npz'
    np.savez(dataset_path, observations=np.zeros((8, 2), dtype=np.float32))
    dataset_id = _expected_dataset_id(env_name, dataset_path)
    transitions = 7
    terminals = np.zeros(transitions, dtype=bool)
    terminals[-1] = True
    valid = _chunk_valid_for_test(terminals, horizon=3)
    source = np.where(valid, 1, 3).astype(np.uint8)
    target_set = TargetSet(
        y=np.where(valid, 0.7, 0.0).astype(np.float32), valid=valid,
        h=3, unit='steps', graph_hash='a' * 64, dataset_id=dataset_id,
        source=source, kind='gamma_pow', loss_mask=valid.copy(),
        discount_gamma=0.9,
    )
    target_path = tmp_path / 'targets.npz'
    save_targets(target_set, target_path)

    compact = {
        'observations': np.zeros((transitions + 1, 2), dtype=np.float32),
        'actions': np.zeros((transitions + 1, 1), dtype=np.float32),
        'terminals': np.r_[terminals, True].astype(np.float32),
        'valids': np.r_[np.ones(transitions), 0.0].astype(np.float32),
    }
    attached = _attach_graph_targets(
        compact, target_path, dataset_id, env_name, 3, 0.9,
        dataset_path=dataset_path,
    )
    assert attached.kind == 'gamma_pow'
    np.testing.assert_array_equal(compact['graph_target'][:transitions], target_set.y)
    assert compact['graph_target'][-1] == 0.0

    with pytest.raises(ValueError, match='horizon'):
        _attach_graph_targets(
            compact, target_path, dataset_id, env_name, 5, 0.9,
            dataset_path=dataset_path,
        )
    with pytest.raises(ValueError, match='active OGBench dataset archive'):
        _attach_graph_targets(
            compact, target_path, 'wrong-id', env_name, 3, 0.9,
            dataset_path=dataset_path,
        )


def _chunk_valid_for_test(terminals, horizon):
    valid = np.zeros(len(terminals), dtype=bool)
    n_starts = max(0, len(terminals) - horizon + 1)
    if n_starts:
        prefix = np.r_[0, np.cumsum(terminals, dtype=np.int64)]
        starts = np.arange(n_starts)
        valid[:n_starts] = (prefix[starts + horizon - 1] - prefix[starts]) == 0
    return valid
