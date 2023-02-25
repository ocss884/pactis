import torch
from torch import nn
from typing import List, Dict
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


# class TimeSeriesOM:
#     r"""
#     The main class for recoding the missing time steps' index with their nearest observations and indicating 
#     the next missing time steps to be sampled from their nearest observations. 
    
#     e.g. A time series with 10 time steps, and the time index=(1,2,4,7) are missing.
#     index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     mask  = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1]
#     The nearest observations of missing time steps are:
#     1: (0, 3); 2: (0, 3); 4: (3, 5); 7: (6, 8)
#     """
#     def __init__(self, mask: List) -> None:
#         self._ori_mask = mask
#         self._mask = mask
#         self._nearst_observation_map = OrderedDict()
        
#         self._fill_if_ends_missing()
#         self._init_observation_map()
            
#     @property
#     def nearst_observation_map(self):
#         return self._nearst_observation_map
#     @property
#     def original_mask(self):
#         return self._ori_mask
#     @property
#     def current_mask(self):
#         return self._mask
#     @property
#     def current_history(self):
#         return [id for id, value in enumerate(self.current_mask) if value == 1]

#     def _fill_if_ends_missing(self) -> None:
#         # at least one end none-zero
#         if self._mask[0] == 0:
#             # use mask[-1] to fill the first missing
#             self._mask[0] = 1
#         if self._mask[-1] == 0:
#             # use mask[0] to fill the last missing
#             self._mask[-1] = 1

#     def _init_observation_map(self) -> None:
#         self._nearst_observation_map = self.find_all_missing()
    
#     def _update_once(self, left_pivot, right_pivot, observation_map, mid_points_map):
#         left_pivot -= 1; right_pivot += 1
#         # when num of missing time steps is even, the mid point is the left one
#         mid_point = int((left_pivot + right_pivot)/2) 
#         mid_points_map[mid_point] = observation_map.pop(mid_point)
#         for key, value in observation_map.items():
#             if value[0] == left_pivot and value[1] == right_pivot:
#                 if key < mid_point:
#                     observation_map[key][1] = mid_point
#                 else:
#                     observation_map[key][0] = mid_point
#         self.current_mask[mid_point] = 1

#         return mid_points_map

#     def has_missing_points(self) -> bool:
#         return (0 in self.current_mask)
    
#     def find_all_missing(self) -> Dict[int, Dict[str, int]]:
#         # O(n)
#         left_pivot = 0
#         # {missing point index: {"left": id_left, "right": id_right}}
#         # contains all missing points index with their nearest left and right observations (value=1)
#         nearest_observation_map = OrderedDict()
#         for idx, value in enumerate(self.current_mask):
#             if value == 1:
#                 if idx-left_pivot > 1:
#                     # there are missing points between left_pivot and idx
#                     for i in range(left_pivot+1, idx):
#                         # nearest_observation_map[i] = {"left": left_pivot, "right": idx}
#                         nearest_observation_map[i] = [left_pivot, idx]
#                 left_pivot = idx
#         return nearest_observation_map
    
#     def next_time_steps_to_sample(self) -> Dict[int, Dict[str, int]]:
#         r"""
#         Return the current mid points and update the current map
#         Iterate over Missing PoinTs but not the entire series to reduce the time complexity
#         """
#         first_idx = next(iter(self._nearst_observation_map))
#         left_mpt = first_idx
#         last_mpt = first_idx
        
#         map_copy = self._nearst_observation_map.copy()
#         mid_points_map = {}
        
#         for idx in self._nearst_observation_map:
#             if idx-last_mpt > 1:
#                 self._update_once(left_mpt, last_mpt, map_copy, mid_points_map)
#                 left_mpt = idx
#             last_mpt = idx
        
#         self._update_once(left_mpt, last_mpt, map_copy, mid_points_map)
#         self._nearst_observation_map = map_copy
#         return mid_points_map