import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from robot_pg.rl.ppo import PolicyNet
from robot_pg.rl.train import checkpoint_metadata, load_policy, main


def test_train_rejects_nonpositive_steps(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["--output", str(tmp_path / "x.pt"), "--steps", "0"])
    assert exit_info.value.code == 2
    assert not (tmp_path / "x.pt").exists()


def test_checkpoint_metadata_loads_with_weights_only(tmp_path):
    metadata = checkpoint_metadata(7, 100)
    saved = PolicyNet(4, 1)
    path = tmp_path / "meta.pt"
    torch.save({"state_dict": saved.state_dict(), "metadata": metadata}, path)
    net, loaded = load_policy(path)
    assert {k: v for k, v in loaded.items() if not k.endswith("_version")} == {
        "robot": "cartpole_playground", "joint_names": ["cart", "pole"], "actuator_names": ["force"],
        "control_dt": 0.02, "observation_size": 4, "action_size": 1, "hidden_sizes": [64, 64],
        "activation": "tanh", "seed": 7, "total_steps": 100,
    }
    obs = torch.tensor([[0.1, -0.2, 0.3, -0.4]])
    assert torch.equal(net(obs)[0], saved(obs)[0])


def test_short_train_writes_a_checkpoint(tmp_path):
    path = tmp_path / "short.pt"
    assert main(["--output", str(path), "--steps", "64", "--seed", "3"]) == 0
    net, metadata = load_policy(path)
    assert (metadata["seed"], metadata["total_steps"]) == (3, 64)
    assert net.log_std.shape == (1,)
