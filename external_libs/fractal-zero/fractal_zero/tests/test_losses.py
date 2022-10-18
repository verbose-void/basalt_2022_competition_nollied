import gym
import gym.spaces as spaces

import torch
import torch.nn.functional as F

from fractal_zero.loss.space_loss import (
    BoxSpaceLoss,
    DictSpaceLoss,
    DiscreteSpaceLoss,
    SpaceLoss,
)


def _general_assertions(space: spaces.Space, criterion: SpaceLoss):

    # non-batched
    for _ in range(10):
        x0 = space.sample()
        x1 = space.sample()
        print(x0, x1)
        criterion(x0, x1)

    # batched
    for _ in range(10):
        x0 = [space.sample() for _ in range(10)]
        x1 = [space.sample() for _ in range(10)]
        criterion(x0, x1)

    # TODO: out of range warning?


def test_box():
    space = spaces.Box(low=0, high=2, shape=(5, 3))
    criterion = BoxSpaceLoss(space)

    _general_assertions(space, criterion)


def test_discrete():
    space = spaces.Discrete(5)
    criterion = DiscreteSpaceLoss(space)
    _general_assertions(space, criterion)

    # use cross entropy (expects logits)
    criterion = DiscreteSpaceLoss(space, loss_func=F.cross_entropy)

    # no batch
    logits = torch.tensor([0, 0.1, 0.5, 1, 1], requires_grad=True)
    target = space.sample()
    loss = criterion(logits, target)
    print("loss", loss)
    assert loss.shape == tuple()
    loss.backward()

    # batch
    blogits = torch.tensor(
        [[0, 0.1, 0.5, 1, 1], [0, 0.1, 0.5, 1, 1], [0, 0.1, 0.5, 1, 1]],
        requires_grad=True,
    )
    btarget = [space.sample() for _ in range(3)]
    bloss = criterion(blogits, btarget)
    print("bloss", bloss)
    assert bloss.shape == tuple()
    bloss.backward()


def test_discrete_dict():
    space = spaces.Discrete(5)

    # define a DictLoss with different loss functions for Discrete spaces.
    criterion = DictSpaceLoss(
        spaces.Dict(
            {
                "space0": space,
                "subspace": spaces.Dict({"space1": space}),
            }
        ),
        loss_spec={
            "space0": F.mse_loss,  # if this was None, should be default
            "subspace": {"space1": F.cross_entropy},
        },
    )

    space0_sample = torch.tensor([4.0], requires_grad=True)
    space0_target = torch.tensor([10.0])

    space1_sample = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.3, 0.2]], requires_grad=True)
    space1_target = torch.tensor([2, 1])

    expected_loss = F.mse_loss(space0_sample, space0_target) + F.cross_entropy(
        space1_sample, space1_target
    )

    y = {"space0": space0_sample, "subspace": {"space1": space1_sample}}
    t = {"space0": space0_target, "subspace": {"space1": space1_target}}
    actual_loss = criterion(y, t)
    actual_loss.backward()

    print(expected_loss, actual_loss)
    assert torch.isclose(expected_loss, actual_loss)


def test_dict():
    space = gym.spaces.Dict(
        {
            "x": gym.spaces.Discrete(4),
            "y": gym.spaces.Box(low=0, high=1, shape=(2,)),
        }
    )

    space.seed(5)
    criterion = DictSpaceLoss(space)

    a0 = {
        "x": torch.tensor(3.0, requires_grad=True),
        "y": torch.tensor([0.3, 0.1], requires_grad=True),
    }
    a1 = space.sample()

    loss = criterion(a0, a1)

    print(a0)
    print(a1)
    print(loss)

    loss.backward()
    assert torch.isclose(loss, torch.tensor(1.0553), rtol=0.0001)

    # with batch
    a0_batch = [a0 for _ in range(5)]
    a1_batch = [space.sample() for _ in range(5)]

    bloss = criterion(a0_batch, a1_batch)
    bloss.backward()
    assert torch.isclose(bloss, torch.tensor(3.3352), rtol=0.0001)
