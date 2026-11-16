import pytest

torch = pytest.importorskip("torch")

from robot_pg.rl.ppo import (
    GAMMA,
    LR,
    MINIBATCH_SIZE,
    PolicyNet,
    ValueNet,
    gae,
    ppo_update,
)


def test_gae_terminates_without_bootstrap():
    adv, ret = gae(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([True]),
                   torch.tensor([False]), torch.tensor([99.0]))
    assert adv.item() == pytest.approx(1.0)
    assert ret.item() == pytest.approx(1.0)


def test_gae_truncation_bootstraps_next_value():
    adv, ret = gae(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([False]),
                   torch.tensor([True]), torch.tensor([2.0]))
    assert adv.item() == pytest.approx(1.0 + GAMMA * 2.0)
    assert ret.item() == pytest.approx(1.0 + GAMMA * 2.0)


def test_gae_masks_next_advantage_before_terminal_step():
    adv, ret = gae(
        torch.tensor([1.0, 10.0, 100.0]), torch.zeros(3), torch.tensor([False, True, False]),
        torch.tensor([False, False, False]), torch.tensor([0.0, 99.0, 0.0]), gamma=1.0, lam=1.0,
    )
    assert adv.tolist() == pytest.approx([11.0, 10.0, 100.0])
    assert ret.tolist() == pytest.approx([11.0, 10.0, 100.0])


def test_gae_truncation_bootstraps_then_masks():
    adv, ret = gae(
        torch.tensor([1.0, 1.0, 50.0]), torch.zeros(3), torch.tensor([False, False, False]),
        torch.tensor([False, True, False]), torch.tensor([0.0, 2.0, 0.0]), gamma=1.0, lam=1.0,
    )
    assert adv.tolist() == pytest.approx([4.0, 3.0, 50.0])
    assert ret.tolist() == pytest.approx([4.0, 3.0, 50.0])


Setup = tuple[PolicyNet, ValueNet, "torch.optim.Optimizer", "torch.Tensor", "torch.Tensor", "torch.Tensor",
              "torch.Tensor", "torch.Tensor"]


def update_setup(t: int) -> Setup:
    torch.manual_seed(0)
    policy, value_net = PolicyNet(4, 2), ValueNet(4)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=LR)
    observations, actions = torch.randn(t, 4), torch.randn(t, 2)
    with torch.no_grad():
        mean, log_std = policy(observations)
        old_log_probs = torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(-1)
        values = value_net(observations).reshape(t)
    advantages = torch.randn(t)
    return policy, value_net, optimizer, observations, actions, old_log_probs, advantages, values + advantages


def test_ppo_update_changes_the_policy_with_a_partial_minibatch():
    setup = update_setup(MINIBATCH_SIZE + 1)
    assert setup[5].shape == (MINIBATCH_SIZE + 1,)
    first = next(setup[0].parameters()).detach().clone()
    ppo_update(*setup)
    assert not torch.equal(first, next(setup[0].parameters()).detach())


def test_ppo_update_is_invariant_to_advantage_scale():
    a, b = update_setup(2 * MINIBATCH_SIZE + 1), update_setup(2 * MINIBATCH_SIZE + 1)
    torch.manual_seed(123)
    ppo_update(*a)
    torch.manual_seed(123)
    ppo_update(*b[:6], 100.0 * b[6] + 5.0, b[7])
    for pa, pb in zip(a[0].parameters(), b[0].parameters(), strict=True):
        assert torch.allclose(pa, pb, atol=1e-5)


def test_ppo_update_single_sample_minibatch_is_finite():
    setup = update_setup(1)
    ppo_update(*setup)
    for p in [*setup[0].parameters(), *setup[1].parameters()]:
        assert torch.isfinite(p).all()


def test_policy_starts_with_unit_std_and_v0_parameter_names():
    net = PolicyNet(4, 1)
    _, log_std = net(torch.zeros(3, 4))
    assert log_std.tolist() == [0.0]
    assert list(net.state_dict()) == [
        "log_std", "mean_head.0.weight", "mean_head.0.bias", "mean_head.2.weight", "mean_head.2.bias",
        "mean_head.4.weight", "mean_head.4.bias",
    ]
