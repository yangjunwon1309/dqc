import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))

class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

        (self.terminal_locs,) = np.nonzero(self['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        if 'next_actions' not in result:
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    
    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. They are
        stored in the key 'value_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        
        if 'oracle_reps' in self.dataset:
            batch['value_goals'] = self.dataset['oracle_reps'][value_goal_idxs]
        else:
            batch['value_goals'] = self.get_observations(value_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        if discount is None:
            discount = self.config['discount']

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - discount, size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs
    
    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])

@dataclasses.dataclass
class CGCDataset(GCDataset):
    """Dataset class for action-chunked goal-conditioned RL.

    This class extends GCDataset to support action chunks. It reads the following additional key from the config:
    - backup_horizon: Subgoal steps.
    """

    def __post_init__(self):
        super().__post_init__()
        # Set valid_idxs for action chunks.
        cur_idx = 0
        valid_idxs = []
        
        for terminal_idx in self.terminal_locs:
            valid_idxs.append(np.arange(cur_idx, terminal_idx + 1 - self.config['backup_horizon']))
            cur_idx = terminal_idx + 1
        self.dataset.valid_idxs = np.concatenate(valid_idxs)

    def compute_high_next_idxs(self, idxs, final_state_idxs, high_goal_idxs, backup_horizon):
        """Compute the next indices for high-level goals."""
        batch_size = len(idxs)
        backup_horizon = np.full(batch_size, backup_horizon)

        # Clip to the end of the trajectory.
        backup_horizon = np.minimum(backup_horizon, final_state_idxs - idxs)

        # assert np.all(backup_horizon == self.config['backup_horizon'])

        # Clip to the high-level goal.
        diff_idxs = high_goal_idxs - idxs
        should_clip = (0 <= diff_idxs) & (diff_idxs < backup_horizon)
        backup_horizon = np.where(should_clip, diff_idxs, backup_horizon)

        return idxs + backup_horizon, backup_horizon

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # Sample high-level value goals.
        high_value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        value_backup_horizon = self.config['backup_horizon'] if self.config.get('value_backup_horizon') is None else self.config['value_backup_horizon']
        high_value_next_idxs, high_value_backup_horizon = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_value_goal_idxs,
            value_backup_horizon,
        )

        if 'oracle_reps' in self.dataset:
            batch['high_value_reps'] = self.dataset['oracle_reps'][idxs]
            batch['high_value_goals'] = self.dataset['oracle_reps'][high_value_goal_idxs]
            batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        else:
            batch['high_value_reps'] = batch['observations']
            batch['high_value_goals'] = self.get_observations(high_value_goal_idxs)
            batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        
        batch['high_value_offsets'] = high_value_goal_idxs - idxs
        chunk_offsets = np.arange(self.config['backup_horizon'])
        chunk_idxs = np.minimum(idxs[:, None] + chunk_offsets, final_state_idxs[:, None])
        batch["valids"] = (idxs[:, None] + chunk_offsets <= final_state_idxs[:, None]).astype(float)
        batch['high_value_action_chunks'] = self.dataset['actions'][chunk_idxs].reshape(batch_size, -1)

        high_value_successes = (high_value_backup_horizon < value_backup_horizon).astype(float)
        batch['high_value_backup_horizon'] = high_value_backup_horizon
        batch['high_value_masks'] = 1.0 - high_value_successes
        batch['high_value_rewards'] = (self.config['discount'] ** high_value_backup_horizon) * high_value_successes

        # One-step info.
        successes = (idxs == high_value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes

        return batch


@dataclasses.dataclass
class CSingleDataset:
    """Unconditional action-chunk dataset for fixed-task sparse puzzles.

    Compact OGBench rows retain terminal-only observations. TD mode follows
    CGCDataset's conservative full-chunk boundary rule. Graph mode follows the
    target artifact's rule, which permits a terminal on the last action; the
    retained terminal-only observation supplies that chunk's endpoint. Neither
    mode samples a chunk that crosses an episode boundary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        self.terminal_locs = np.flatnonzero(np.asarray(self.dataset['terminals']) > 0)
        if not len(self.terminal_locs) or self.terminal_locs[-1] != self.size - 1:
            raise ValueError('single-task compact dataset must end on a terminal row')
        horizon = int(self.config['backup_horizon'])
        current = 0
        valid = []
        for terminal in self.terminal_locs:
            valid.append(np.arange(current, int(terminal) + 1 - horizon, dtype=np.int64))
            current = int(terminal) + 1
        valid = [part for part in valid if len(part)]
        dqc_valid_idxs = np.concatenate(valid) if valid else np.empty(0, dtype=np.int64)
        if self.config.get('critic_mode', 'td') == 'graph':
            if not all(key in self.dataset for key in (
                'graph_target', 'graph_target_valid', 'graph_target_loss_mask'
            )):
                raise ValueError('graph-mode single-task data is missing graph targets')
            transition_rows = np.asarray(self.dataset['valids']) > 0
            graph_valid = (
                np.asarray(self.dataset['graph_target_valid'], dtype=bool)
                & np.asarray(self.dataset['graph_target_loss_mask'], dtype=bool)
                & transition_rows
            )
            # The graph target convention allows a terminal on the last action
            # of a full chunk. Compact data keeps that terminal next-state row,
            # so the endpoint remains available without crossing episodes.
            self.dataset.valid_idxs = np.flatnonzero(graph_valid)
        else:
            self.dataset.valid_idxs = dqc_valid_idxs
        if not len(self.dataset.valid_idxs):
            raise ValueError('dataset has no complete action chunks for backup_horizon')
        if 'masks' not in self.dataset or 'button_states' not in self.dataset:
            raise ValueError('single-task puzzle data requires OGBench masks and button_states')

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        idxs = np.asarray(idxs, dtype=np.int64)
        batch = self.dataset.sample(batch_size, idxs)
        horizon = int(self.config['backup_horizon'])
        offsets = np.arange(horizon, dtype=np.int64)
        action_idxs = idxs[:, None] + offsets[None, :]
        batch['high_value_action_chunks'] = self.dataset['actions'][action_idxs].reshape(
            len(idxs), -1
        )
        batch['high_value_next_observations'] = self.get_observations(idxs + horizon)
        batch['high_value_goals'] = None
        batch['high_value_offsets'] = np.full(len(idxs), horizon, dtype=np.int64)
        batch['valids'] = np.ones((len(idxs), horizon), dtype=np.float32)

        # OGBench's relabeler marks whether each compact state already meets
        # the fixed puzzle goal. Label first success at s[t+k] by gamma**k.
        state_success = np.asarray(self.dataset['masks']) == 0
        future = state_success[idxs[:, None] + np.arange(1, horizon + 1)[None, :]]
        has_success = np.any(future, axis=1)
        first_k = np.argmax(future, axis=1).astype(np.int32) + 1
        starts_success = state_success[idxs]
        success = has_success | starts_success
        k = np.where(starts_success, 0, first_k)
        batch['high_value_backup_horizon'] = np.where(success, k, horizon).astype(np.int32)
        batch['high_value_masks'] = (~success).astype(np.float32)
        batch['high_value_rewards'] = np.where(
            success, np.power(float(self.config['discount']), k), 0.0
        ).astype(np.float32)

        if 'graph_target' in self.dataset:
            for key in ('graph_target', 'graph_target_valid', 'graph_target_loss_mask'):
                batch[key] = self.dataset[key][idxs]
        return batch

    def get_observations(self, idxs):
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])


def make_task_dataset(task_mode, dataset, config, goal_dataset_class=CGCDataset):
    """Select DQC's original goal path or its fixed-task single path."""
    if task_mode == 'goal':
        return goal_dataset_class(dataset, config)
    if task_mode == 'single':
        return CSingleDataset(dataset, config)
    raise ValueError("task_mode must be 'goal' or 'single'")
