import torch
import numpy as np
from torch import nn
from typing import Sequence, Dict
from collections import OrderedDict


class Sequential(nn.Sequential):
    def forward(self, *x, **kwargs):
        for i, module in enumerate(self):
            if type(x) == tuple:
                if i == 0:
                    x = module(*x, **kwargs)
                else:
                    x = module(*x)
            else:
                x = module(x)
        return x


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) + args[0]


def _easy_mlp(
        input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, activation: nn.Module) -> nn.Sequential:
    """
    Generate a MLP with the given parameters.
    """

    elayers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(1, num_layers):
        elayers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    elayers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*elayers)


class TimeSeriesOM:
    r"""
    The main class for recoding the missing time steps' index with their nearest observations and indicating
    the next missing time steps to be sampled from their nearest observations.

    e.g. A time series with 10 time steps, and the time index=(1,2,4,7) are missing.
    index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mask  = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1]
    The nearest observations of missing time steps are:
    1: [0, 3]; 2: [0, 3]; 4: [3, 5]; 7: [6, 8]
    if series > 1,
    """

    def __init__(self, num_series: int, mask: torch.Tensor) -> None:
        self.num_series = num_series
        self.num_time_steps = len(mask)
        self.variables = num_series * self.num_time_steps
        # if torch.is_tensor(mask):
        #     self._mask = torch.clone(mask)
        # else:
        
        self._mask = mask.clone() if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
        self._ori_mask = mask
        self._nearst_observation_map = OrderedDict()

        # self._fill_if_ends_missing()
        self._init_observation_map()

    @property
    def nearst_observation_map(self):
        return self._nearst_observation_map

    @property
    def original_mask(self):
        return self._ori_mask

    @property
    def current_mask(self):
        return self._mask

    def _has_ends_missing(self) -> bool:
        return (0 in self.nearst_observation_map) or (self.num_time_steps-1 in self.nearst_observation_map)

    def _fill_missing_end_points(self) -> Dict | None:
        if (0 not in self.nearst_observation_map) and (self.num_time_steps-1 not in self.nearst_observation_map):
            return

        mid_point_map = {}
        if 0 in self.nearst_observation_map:
            first = self.nearst_observation_map[0][0]
            for n, i in enumerate(range(0, self.variables, self.num_time_steps)):
                ls = np.zeros(2*self.num_series, dtype=int)
                ls[:self.num_series] = np.arange(first, self.variables, self.num_time_steps) if n == 0 else \
                                       np.concatenate((np.arange(0, n*self.num_time_steps, self.num_time_steps),
                                                       np.arange(first+n*self.num_time_steps, self.variables, self.num_time_steps)))
                mid_point_map[i] = ls
                # if n == 0:
                #     mid_point_map[i] = ls
                # else:
                #     mid_point_map[i] = np.concatenate((np.arange(0, n*self.num_time_steps, self.num_time_steps),
                #                                        np.arange(first+n*self.num_time_steps, self.variables, self.num_time_steps)))

            self.current_mask[0] = 1
            self.nearst_observation_map.pop(0)

        if self.num_time_steps-1 in self.nearst_observation_map:
            last = self.nearst_observation_map[self.num_time_steps-1][0]
            for n, i in enumerate(range(self.num_time_steps-1, self.variables, self.num_time_steps)):
                # if n == 0:
                #     mid_point_map[i] = np.arange(
                #         last, self.variables, self.num_time_steps)
                # else:
                #     mid_point_map[i] = np.concatenate((np.arange(self.num_time_steps-1, n*self.num_time_steps, self.num_time_steps),
                #                                        np.arange(last + n*self.num_time_steps, self.variables, self.num_time_steps)))

                ls = np.zeros(2*self.num_series, dtype=int)
                ls[:self.num_series] = np.arange(last, self.variables, self.num_time_steps) if n == 0 else \
                                       np.concatenate((np.arange(self.num_time_steps-1, n*self.num_time_steps, self.num_time_steps),
                                                       np.arange(last + n*self.num_time_steps, self.variables, self.num_time_steps)))
                mid_point_map[i] = ls

            self.current_mask[-1] = 1
            self.nearst_observation_map.pop(self.num_time_steps-1)

        return mid_point_map

    def _init_observation_map(self) -> None:
        self._nearst_observation_map = self.find_all_missing()

    def _update_once(self, left_pivot, right_pivot, observation_map, mid_points_map):
        left_pivot -= 1
        right_pivot += 1
        # when num of missing time steps is even, the mid point is the left one
        mid_point = int((left_pivot + right_pivot)/2)
        l_nbr, r_nbr = observation_map.pop(mid_point)

        for n, i in enumerate(range(mid_point, self.variables, self.num_time_steps)):
            if n == 0:
                mid_points_map[i] = np.concatenate(
                    (np.arange(l_nbr, self.variables, self.num_time_steps),
                     np.arange(r_nbr, self.variables, self.num_time_steps))
                )
            else:
                mid_points_map[i] = np.concatenate(
                    (np.arange(l_nbr, self.variables, self.num_time_steps),
                     np.concatenate(
                        (np.arange(mid_point, n*self.num_time_steps, self.num_time_steps),
                         np.arange(r_nbr+n*self.num_time_steps,
                                   self.variables, self.num_time_steps)
                         )))
                )
        for key, value in observation_map.items():
            if value[0] == left_pivot and value[1] == right_pivot:
                if key < mid_point:
                    observation_map[key][1] = mid_point
                else:
                    observation_map[key][0] = mid_point
        self.current_mask[mid_point] = 1

        return mid_points_map

    def has_missing_points(self) -> bool:
        return 0 in self.current_mask

    def find_all_missing(self) -> Dict[int, Dict[str, int]]:
        # O(n)
        # The return dictionary behaves as two ends are observed even they are not. Since we're always going to
        # check & fill the end points first, it doesn't matter.
        nearest_observation_map = {}

        if self.current_mask[0] == 0 or self.current_mask[-1] == 0:
            # first: index of first observed step
            # last: ~ last observed step
            first, last = -1, -1
            for idx, value in enumerate(self.current_mask):
                if value != 1:
                    continue
                if first == -1:
                    first = idx
                last = idx
            if self.current_mask[0] == 0:
                nearest_observation_map[0] = [first]
                self.current_mask[0] = 1
            if self.current_mask[-1] == 0:
                nearest_observation_map[self.num_time_steps-1] = [last]
                self.current_mask[-1] = 1

        left_pivot = 0
        # {missing point index: [id_left, id_right]}
        # contains all missing points index with their nearest left and right observations (value=1)
        # assume we have two ends observed
        for idx, value in enumerate(self.current_mask): 
            if value == 1:
                if idx-left_pivot > 1:
                    # there are missing points between left_pivot and idx
                    for i in range(left_pivot+1, idx):
                        # nearest_observation_map[i] = {"left": left_pivot, "right": idx}
                        nearest_observation_map[i] = [left_pivot, idx]
                left_pivot = idx
        return nearest_observation_map

    def next_to_fill(self) -> Dict[int, Dict[str, int]]:
        r"""
        Return the current mid points and update the current map
        Iterate over Missing PoinTs but not the entire series to reduce the time complexity
        """
        # print(self._nearst_observation_map)
        if self._has_ends_missing():
            mid_points_map = self._fill_missing_end_points()
            # index_mask = dict.fromkeys(mid_points_map.keys(), torch.as_tensor([1]*self.num_series + [0]*self.num_series, dtype=bool))
            return mid_points_map

        first_idx = next(iter(self._nearst_observation_map))
        left_mpt = first_idx
        last_mpt = first_idx

        map_copy = self._nearst_observation_map.copy()
        mid_points_map = {}

        for idx in self._nearst_observation_map:
            if idx-last_mpt > 1:
                self._update_once(left_mpt, last_mpt, map_copy, mid_points_map)
                left_mpt = idx
            last_mpt = idx

        self._update_once(left_mpt, last_mpt, map_copy, mid_points_map)
        self._nearst_observation_map = map_copy

        # index_mask = dict.fromkeys(mid_points_map.keys(), torch.as_tensor([1]*(self.num_series*2)))
        return mid_points_map

# if __name__ == "__main__":
#     # print(TimeSeriesOM.__doc__)
#     mask = [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0] #14
#     seq = TimeSeriesOM(3, mask)
#     mid_points_map = {}
#     index_mask = {}
#     while seq.has_missing_points():
#         # print(seq.current_mask, "\n")
#         current_mid_points_map, current_index_mask = seq.next_to_fill()
#         mid_points_map, index_mask= {**mid_points_map, **current_mid_points_map}, {**index_mask, **current_index_mask}
#         print(mid_points_map)
#         print(np.asarray(list(current_mid_points_map.values())).reshape(-1, 3, 6).transpose(1, 0, 2).shape)
#         print("-------")


