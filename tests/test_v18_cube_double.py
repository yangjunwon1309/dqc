import numpy as np

from utils.datasets import CSingleDataset, Dataset


def test_cube_double_single_task_graph_chunks_do_not_require_puzzle_button_sidecar():
    n_transitions = 12
    n_rows = n_transitions + 1
    terminals = np.zeros(n_rows, dtype=np.float32)
    terminals[-2:] = 1.0  # compact transition terminal and terminal-only state
    valids = np.ones(n_rows, dtype=np.float32)
    valids[-1] = 0.0
    target_valid = np.zeros(n_transitions, dtype=bool)
    target_valid[: n_transitions - 5 + 1] = True
    fields = {
        "observations": np.zeros((n_rows, 37), dtype=np.float32),
        "actions": np.zeros((n_rows, 5), dtype=np.float32),
        "terminals": terminals,
        "valids": valids,
        "masks": np.ones(n_rows, dtype=np.float32),
    }
    for name, values in (
        ("graph_target", np.full(n_transitions, 0.5, dtype=np.float32)),
        ("graph_target_valid", target_valid),
        ("graph_target_loss_mask", target_valid.copy()),
    ):
        aligned = np.zeros(n_rows, dtype=values.dtype)
        aligned[:n_transitions] = values
        fields[name] = aligned

    dataset = CSingleDataset(
        Dataset.create(**fields),
        {"backup_horizon": 5, "discount": 0.99, "critic_mode": "graph"},
    )
    batch = dataset.sample(2, idxs=np.asarray([0, 1], dtype=np.int64))
    assert batch["high_value_action_chunks"].shape == (2, 25)
    assert batch["graph_target"].shape == (2,)
