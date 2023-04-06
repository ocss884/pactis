"""
>> Compatibility shells between the TACTiS models and the GluonTS and PyTorchTS libraries.
"""

from typing import Any, Dict

import torch
from torch import nn

from .pactis import PACTiS


class PACTiSTrainingNetwork(nn.Module):
    """
    A shell on top of the PACTiS module, to be used during training only.
    """

    def __init__(
        self,
        num_series: int,
        model_parameters: Dict[str, Any],
    ):
        """
        Parameters:
        -----------
        num_series: int
            Number of series of the data which will be sent to the model.
        model_parameters: Dict[str, Any]
            The parameters of the underlying TACTiS model, as a dictionary.
        """
        super().__init__()

        self.model = PACTiS(num_series, **model_parameters)

    def forward(
        self,
        past_target_norm: torch.Tensor,
        future_target_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        past_target_norm: torch.Tensor [batch, time steps, series]
            The historical data that will be available at inference time.
        future_target_norm: torch.Tensor [batch, time steps, series]
            The data to be forecasted at inference time.

        Returns:
        --------
        loss: torch.Tensor []
            The loss function, averaged over all batches.
        """
        # The data coming from Gluon is not in the shape we use in the model, so transpose it.
        device = past_target_norm.device
        hist_value = past_target_norm.transpose(1, 2)
        pred_value = future_target_norm.transpose(1, 2)
        value = torch.cat((hist_value, pred_value), dim=2)
        
        missing_number = pred_value.shape[2]
        num_time_steps = value.shape[2]
        mask = torch.ones(num_time_steps, dtype=torch.bool, device=device)
        # mask[torch.randperm(num_time_steps)[:missing_number]] = False
        mask[-missing_number:] = False
        # For the time steps, we take for granted that the data is aligned with a constant frequency
        # hist_time = torch.arange(0, hist_value.shape[2], dtype=int, device=hist_value.device)[None, :].expand(
        #     hist_value.shape[0], -1
        # )
        # pred_time = torch.arange(
        #     hist_value.shape[2], hist_value.shape[2] + pred_value.shape[2], dtype=int, device=pred_value.device
        # )[None, :].expand(pred_value.shape[0], -1)

        return self.model.loss(value, mask)


class PACTiSPredictionNetwork(nn.Module):
    """
    A shell on top of the PACTiS module, to be used during inference only.
    """

    def __init__(
        self,
        num_series: int,
        model_parameters: Dict[str, Any],
        prediction_length: int,
        num_parallel_samples: int,
    ):
        """
        Parameters:
        -----------
        num_series: int
            Number of series of the data which will be sent to the model.
        model_parameters: Dict[str, Any]
            The parameters of the underlying TACTiS model, as a dictionary.
        """
        super().__init__()

        self.model = PACTiS(num_series, **model_parameters)
        self.num_parallel_samples = num_parallel_samples
        self.prediction_length = prediction_length

    def forward(
        self,
        past_target_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        past_target_norm: torch.Tensor [batch, time steps, series]
            The historical data that are available.

        Returns:
        --------
        samples: torch.Tensor [samples, batch, time steps, series]
            Samples from the forecasted distribution.
        """
        device = past_target_norm.device
        # The data coming from Gluon is not in the shape we use in the model, so transpose it.
        hist_value = past_target_norm.transpose(1, 2)

        # For the time steps, we take for granted that the data is aligned with a constant frequency
        hist_time = torch.arange(0, hist_value.shape[2], dtype=int, device=hist_value.device)[None, :].expand(
            hist_value.shape[0], -1
        )
        pred_time = torch.arange(hist_value.shape[2], hist_value.shape[2] + self.prediction_length, dtype=int, device=hist_value.device
        )[None, :].expand(hist_value.shape[0], -1)

        value = torch.cat((hist_value, torch.zeros((hist_value.shape[0], hist_value.shape[1], self.prediction_length)).to(device)), dim=2)
        mask = torch.ones(value.shape[2], dtype=torch.bool, device=value.device)
        mask[self.prediction_length:] = False
        samples = self.model.sample(
            num_samples=self.num_parallel_samples, value=value, mask=mask)
        # The model decoder returns both the observed and sampled values, so removed the observed ones.
        # Also, reorder from [batch, series, time steps, samples] to GluonTS expected [batch, samples, time steps, series].
        return samples[:, :, -self.prediction_length :, :].permute((0, 3, 2, 1))
