from abc import ABC
from typing import Dict, Mapping, Sequence
import torch
import torch.nn.functional as F

import gym
import gym.spaces as spaces


def _float_cast(vec):
    if isinstance(vec, torch.Tensor):
        return vec.float()
    return torch.tensor(vec, dtype=float).float()


def _long_cast(vec):
    if isinstance(vec, torch.Tensor):
        return vec.long()
    return torch.tensor(vec, dtype=torch.long).long()


class SpaceLoss(ABC):
    pass


class DiscreteSpaceLoss(SpaceLoss):
    def __init__(self, discrete_space: spaces.Discrete, loss_func=None):

        # TODO: can we auto-determine the kind of loss func?
        # TODO: ie. if the incoming sample is obviously logits, maybe
        # TODO: we use F.cross_entropy automatically?
        self.loss_func = loss_func if loss_func else F.mse_loss

        if not isinstance(discrete_space, spaces.Discrete):
            raise ValueError(f"Expected Discrete space, got {discrete_space}.")

        self.space = discrete_space

    def _cast_x(self, x) -> torch.Tensor:
        return _float_cast(x)

    def _cast_y(self, y) -> torch.Tensor:
        if self.loss_func == F.mse_loss:
            return _float_cast(y)

        elif self.loss_func == F.cross_entropy:
            return _long_cast(y)

        # raise NotImplementedError(f"Cast is not implemented for {self.loss_func}")
        return y

    def __call__(self, x, y):
        x = self._cast_x(x)
        y = self._cast_y(y)

        # NOTE:
        # discrete loss should be defined for the following scenarios:
        # 1. the straight up "distance" between 2 action samples
        #    for example, if a discrete space is sampled and returns `5`,
        #    and it's being compared to another discrete sample `3`,
        #    the "distance" should be `2`.
        # 2. the actual loss between an action sample and some targets?
        #    for example, when a discrete target is `2` and the input is a
        #    probability distribution `[0.2, 0.2, 0.6]`, the loss should be
        #    cross entropy where the `2` is inferred to be a class target.

        return self.loss_func(x, y)


class BoxSpaceLoss(SpaceLoss):
    def __init__(self, box_space: spaces.Box, loss_func=None):
        self.loss_func = loss_func if loss_func else F.mse_loss
        if not isinstance(box_space, spaces.Box):
            raise ValueError(f"Expected Discrete space, got {box_space}.")
        self.space = box_space

    def __call__(self, x, y):
        return self.loss_func(_float_cast(x), _float_cast(y))


LOSS_CLASSES = {
    spaces.Discrete: DiscreteSpaceLoss,
    spaces.Box: BoxSpaceLoss,
}


class DictSpaceLoss(SpaceLoss):
    def __init__(self, dict_space: spaces.Space, loss_spec: Dict = None):
        self.space = dict_space
        self.loss_spec = loss_spec if loss_spec else {}
        self._build_funcs()

    def _build_funcs(self):
        self.funcs = {}
        for key, subspace in self.space.items():
            subspec = self.loss_spec.get(key, None)
            self.funcs[key] = get_space_loss(subspace, subspec)

    def _dict_loss(self, y, t):
        loss = 0
        for key, func in self.funcs.items():
            loss += func(y[key], t[key])
        return loss

    def __call__(self, y, t):
        both_seqs = isinstance(y, Sequence) and isinstance(t, Sequence)
        both_dicts = isinstance(y, Mapping) and isinstance(t, Mapping)
        if not both_seqs and not both_dicts:
            raise TypeError(
                f"Type mismatch. Expected both inputs to be matching. Got: {type(y)} and {type(t)}."
            )

        if both_seqs:

            if len(y) != len(t):
                raise ValueError(
                    f"Expected both inputs to be the same length. Got {len(y)} and {len(t)}."
                )

            total = 0
            c = 0
            for y_sample, t_sample in zip(y, t):
                total += self._dict_loss(y_sample, t_sample)
                c += 1

            # TODO: support reduce function instead of always doing mean?
            return total / c

        return self._dict_loss(y, t)


def get_space_loss(space: spaces.Space, spec: Dict = None) -> SpaceLoss:
    if isinstance(space, spaces.Dict):
        return DictSpaceLoss(space, loss_spec=spec)
    return LOSS_CLASSES[type(space)](space, loss_func=spec)
