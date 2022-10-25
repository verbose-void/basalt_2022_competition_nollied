from copy import deepcopy
from typing import Any, Callable, List, Sequence
import gym
import numpy as np

import torch
import torch.nn.functional as F


def parameters_norm(parameters):
    c = 0
    total = 0
    for param in parameters:
        total += torch.linalg.norm(param)
        c += 1
    return total / c


def dist_of_model_paramters(p0, p1):
    total = 0
    c = 0
    for param0, param1 in zip(p0, p1):
        p0 = param0.data.flatten()
        p1 = param1.data.flatten()
        total += torch.linalg.norm(p0 - p1)  # euclidean distance
        c += 1
    return total / c


def get_space_shape(space):
    if isinstance(space, gym.spaces.Discrete):
        return (1,)

    if isinstance(space, gym.spaces.Box):
        return space.shape

    raise NotImplementedError(f"Type not supported: {type(space)}")


def mean_min_max_dict(name: str, arr) -> dict:
    if isinstance(arr, list):
        arr = torch.tensor(arr, dtype=float)

    return {
        f"{name}/mean": arr.mean(),
        f"{name}/min": arr.min(),
        f"{name}/max": arr.max(),
    }


def _clone_sequence(
    l: Sequence, clone_partners, clone_mask, clone_func: Callable = None
):
    assert len(clone_mask) == len(clone_partners) == len(l)

    new_list = []
    for i in range(len(clone_mask)):
        do_clone = clone_mask[i]
        partner = clone_partners[i]

        if do_clone:
            # NOTE: may not need to deepcopy.
            if clone_func is not None:
                item = clone_func(l[i], l[partner])
            else:
                item = l[partner]
        else:
            item = l[i]

        new_list.append(item)

    assert len(new_list) == len(l)
    return new_list


def cloning_primitive(
    subject: Any, clone_partners, clone_mask, clone_func: Callable = None
):
    if isinstance(subject, (np.ndarray, torch.Tensor)):
        if clone_func is not None:
            raise NotImplementedError("Clone funcs are only defined for sequences.")
        subject[clone_mask] = subject[clone_partners][clone_mask]
        cloned_subject = subject
    elif isinstance(subject, Sequence):
        cloned_subject = _clone_sequence(
            subject, clone_partners, clone_mask, clone_func=clone_func
        )
    else:
        raise NotImplementedError()
    return cloned_subject
