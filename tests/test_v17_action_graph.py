from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.dqc import DQCAgent
from main import _validate_graph_target_level
from utils.datasets import CSingleDataset, Dataset


def _fields(n_transitions=20, horizon=5):
    n_rows = n_transitions + 1
    terminals = np.zeros(n_rows, dtype=np.float32)
    terminals[-2:] = 1.0
    valids = np.ones(n_rows, dtype=np.float32)
    valids[-1] = 0.0
    target_valid = np.zeros(n_transitions, dtype=bool)
    target_valid[:n_transitions - horizon] = True
    fields = {
        'observations': np.arange(n_rows * 3, dtype=np.float32).reshape(n_rows, 3) / 20,
        'actions': np.tanh(np.arange(n_rows, dtype=np.float32).reshape(n_rows, 1) / 10),
        'terminals': terminals,
        'valids': valids,
        'masks': np.ones(n_rows, dtype=np.float32),
        'button_states': np.zeros((n_rows, 9), dtype=np.int64),
    }
    for key, values in (
        ('graph_target', np.linspace(0.2, 0.8, n_transitions, dtype=np.float32)),
        ('graph_target_valid', target_valid),
        ('graph_target_loss_mask', target_valid.copy()),
    ):
        aligned = np.zeros(n_rows, dtype=values.dtype)
        aligned[:n_transitions] = values
        fields[key] = aligned
    return fields


def _agent_config(**overrides):
    config = {
        'lr': 1e-3,
        'value_hidden_dims': (8,),
        'actor_hidden_dims': (8,),
        'layer_norm': False,
        'actor_layer_norm': False,
        'num_qs': 2,
        'use_chunk_critic': False,
        'graph_target_level': 'action',
        'backup_horizon': 5,
        'policy_chunk_size': 5,
        'distill_method': 'expectile',
        'kappa_d': 0.5,
        'implicit_backup_type': 'quantile',
        'kappa_b': 0.9,
        'discount': 0.999,
        'tau': 0.005,
        'q_agg': 'mean',
        'flow_steps': 2,
        'best_of_n': 2,
        'critic_mode': 'graph',
    }
    config.update(overrides)
    return config


def _batch():
    data = Dataset.create(**_fields())
    dataset = CSingleDataset(data, {
        'backup_horizon': 5,
        'discount': 0.999,
        'critic_mode': 'graph',
    })
    return dataset.sample(4, idxs=np.asarray([0, 1, 2, 3], dtype=np.int64))


def _changed(left, right):
    return any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    )


def test_action_graph_loss_only_uses_action_critic_and_leaves_other_modules_unchanged():
    batch = _batch()
    agent = DQCAgent.create(0, batch, _agent_config())
    params_before = agent.network.params

    class GuardNetwork:
        def select(self, name):
            if name in ('chunk_critic', 'value', 'target_action_critic'):
                raise AssertionError(f'action graph loss unexpectedly selected {name}')
            return agent.network.select(name)

    probe = SimpleNamespace(
        config=agent.config, network=GuardNetwork(), bce_loss=DQCAgent.bce_loss
    )
    loss, _ = DQCAgent.action_critic_loss(
        probe, batch, params_before, jax.random.PRNGKey(11)
    )
    assert np.isfinite(float(loss))

    def loss_fn(params):
        return agent.total_loss(batch, params, rng=jax.random.PRNGKey(12))[0]

    grads = jax.grad(loss_fn)(params_before)
    assert 'modules_chunk_critic' not in grads
    assert all(np.all(np.asarray(x) == 0) for x in jax.tree_util.tree_leaves(grads['modules_value']))
    assert all(
        np.all(np.asarray(x) == 0)
        for x in jax.tree_util.tree_leaves(grads['modules_target_action_critic'])
    )

    updated, _ = agent.update(batch)
    params_after = updated.network.params
    assert _changed(params_before['modules_action_critic'], params_after['modules_action_critic'])
    assert not _changed(params_before['modules_value'], params_after['modules_value'])
    assert not _changed(
        params_before['modules_target_action_critic'], params_after['modules_target_action_critic']
    )


def test_action_graph_configuration_rejects_wrong_target_horizon_or_chunk_critic():
    valid = _agent_config()
    _validate_graph_target_level(valid, 'graph', target_h=5)
    with pytest.raises(ValueError, match='target h=5'):
        _validate_graph_target_level(valid, 'graph', target_h=4)
    with pytest.raises(ValueError, match='use_chunk_critic=False'):
        _validate_graph_target_level(
            _agent_config(use_chunk_critic=True), 'graph', target_h=5
        )
    with pytest.raises(ValueError, match='backup_horizon=policy_chunk_size=5'):
        _validate_graph_target_level(
            _agent_config(backup_horizon=4), 'graph', target_h=5
        )


def test_masked_action_graph_rows_do_not_change_the_loss():
    batch = _batch()
    agent = DQCAgent.create(0, batch, _agent_config())
    batch = dict(batch)
    batch['graph_target'] = jnp.asarray([0.2, 0.3, 0.7, 0.9], dtype=jnp.float32)
    batch['graph_target_loss_mask'] = jnp.asarray([1, 0, 1, 0], dtype=jnp.float32)
    loss_a, _ = agent.action_critic_loss(batch, agent.network.params, jax.random.PRNGKey(0))

    changed = dict(batch)
    changed['graph_target'] = jnp.asarray([0.2, 0.99, 0.7, 0.01], dtype=jnp.float32)
    loss_b, _ = agent.action_critic_loss(changed, agent.network.params, jax.random.PRNGKey(0))
    np.testing.assert_allclose(loss_a, loss_b, rtol=0, atol=1e-7)


def _v16_graph_total_loss(agent, batch, params, rng):
    _, actor_rng, action_rng, chunk_rng = jax.random.split(rng, 4)
    config = agent.config

    target_v = jnp.asarray(batch['graph_target'], dtype=jnp.float32)
    chunk_mask = jnp.asarray(batch['graph_target_loss_mask'], dtype=jnp.float32)
    chunk_logit = agent.network.select('chunk_critic')(
        batch['observations'], goals=batch['high_value_goals'],
        actions=batch['high_value_action_chunks'], params=params,
    )
    chunk_per_row = agent.bce_loss(chunk_logit, target_v)
    chunk_mask = jnp.broadcast_to(chunk_mask, chunk_per_row.shape)
    chunk_loss = (chunk_per_row * chunk_mask).sum() / jnp.maximum(chunk_mask.sum(), 1.0)

    # This reproduces the v1.6 graph-mode action-critic expectile distillation.
    _, _ = jax.random.split(action_rng)
    chunk_target = agent.network.select('chunk_critic')(
        batch['observations'], goals=batch['high_value_goals'],
        actions=batch['high_value_action_chunks'],
    )
    target_q = jax.nn.sigmoid(chunk_target)
    action_logit = agent.network.select('action_critic')(
        batch['observations'], goals=batch['high_value_goals'],
        actions=batch['high_value_action_chunks'][..., :config['ac_action_dim']], params=params,
    )
    action_q = jax.nn.sigmoid(action_logit)
    weight = jnp.where(target_q >= action_q, config['kappa_d'], 1 - config['kappa_d'])
    action_loss = (
        weight * agent.bce_loss(action_logit, target_q)
        * batch['valids'][..., config['ac_action_dim'] - 1]
    ).mean()

    batch_size = batch['actions'].shape[0]
    _, x_rng, t_rng, _ = jax.random.split(actor_rng, 4)
    x0 = jax.random.normal(x_rng, (batch_size, config['ac_action_dim']))
    x1 = batch['high_value_action_chunks'][..., :config['ac_action_dim']]
    t = jax.random.uniform(t_rng, (batch_size, 1))
    xt = (1 - t) * x0 + t * x1
    pred = agent.network.select('actor_bc')(
        batch['observations'], actions=xt, times=t, params=params,
    )
    actor_loss = jnp.mean(
        jnp.mean(jnp.square(pred - (x1 - x0)), axis=-1)
        * batch['valids'][..., config['ac_action_dim'] - 1]
    )
    return chunk_loss + action_loss + actor_loss, chunk_loss, action_loss, actor_loss


def test_default_chunk_graph_loss_matches_v16_formula():
    batch = _batch()
    config = _agent_config(
        use_chunk_critic=True,
        graph_target_level='chunk',
        backup_horizon=5,
        policy_chunk_size=1,
    )
    agent = DQCAgent.create(3, batch, config)
    rng = jax.random.PRNGKey(19)
    actual, actual_info = agent.total_loss(batch, agent.network.params, rng=rng)
    expected, chunk_loss, action_loss, actor_loss = _v16_graph_total_loss(
        agent, batch, agent.network.params, rng,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_info['chunk_critic/critic_loss'], chunk_loss, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_info['action_critic/critic_loss'], action_loss, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_info['actor/bc_flow_loss'], actor_loss, rtol=1e-6, atol=1e-7)
