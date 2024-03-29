from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.transform import (
    AddObservedValuesIndicator,
    CDFtoGaussianTransform,
    Chain,
    InstanceSampler,
    InstanceSplitter,
    RenameFields,
    TestSplitSampler,
    Transformation,
    cdf_to_gaussian_forward_transform,
)
from pts import Trainer
from pts.model import PyTorchEstimator
from pts.model.utils import get_module_forward_input_names

from .network import PACTiSPredictionNetwork, PACTiSTrainingNetwork


class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1

        if window_size <= 0:
            return np.array([], dtype=int)

        indices = np.random.randint(window_size, size=1)
        return indices + a


class PACTiSEstimator(PyTorchEstimator):

    def __init__(
        self,
        model_parameters: Dict[str, Any],
        num_series: int,
        history_length: int,
        prediction_length: int,
        trainer: Trainer,
        cdf_normalization: bool = False,
        num_parallel_samples: int = 1,
    ):
        super().__init__(trainer=trainer)

        self.model_parameters = model_parameters

        self.num_series = num_series
        self.history_length = history_length
        self.prediction_length = prediction_length

        self.cdf_normalization = cdf_normalization
        self.num_parallel_samples = num_parallel_samples

    def create_training_network(self, device: torch.device) -> nn.Module:
        """
        Create the encapsulated TACTiS model which can be used for training.

        Parameters:
        -----------
        device: torch.device
            The device where the model parameters should be placed.

        Returns:
        --------
        model: nn.Module
            An instance of TACTiSTrainingNetwork.
        """
        return PACTiSTrainingNetwork(
            num_series=self.num_series,
            model_parameters=self.model_parameters,
        ).to(device=device)

    def create_instance_splitter(self, mode: str) -> Transformation:
        """
        Create and return the instance splitter needed for training, validation or testing.

        Parameters:
        -----------
        mode: str, "training", "validation", or "test"
            Whether to split the data for training, validation, or test (forecast)

        Returns
        -------
        Transformation
            The InstanceSplitter that will be applied entry-wise to datasets,
            at training, validation and inference time based on mode.
        """
        assert mode in ["training", "validation", "test"]

        if mode == "training":
            instance_sampler = SingleInstanceSampler(
                min_past=self.history_length,  # Will not pick incomplete sequences
                min_future=self.prediction_length,
            )
        elif mode == "validation":
            instance_sampler = SingleInstanceSampler(
                min_past=self.history_length,  # Will not pick incomplete sequences
                min_future=self.prediction_length,
            )
        elif mode == "test":
            # This splitter takes the last valid window from each multivariate series,
            # so any multi-window split must be done in the data definition.
            instance_sampler = TestSplitSampler()

        if self.cdf_normalization:
            normalize_transform = CDFtoGaussianTransform(
                cdf_suffix="_norm",
                target_field=FieldName.TARGET,
                target_dim=self.num_series,
                max_context_length=self.history_length,
                observed_values_field=FieldName.OBSERVED_VALUES,
            )
        else:
            normalize_transform = RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_norm",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_norm",
                }
            )

        return (
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self.history_length,
                future_length=self.prediction_length,
                time_series_fields=[FieldName.OBSERVED_VALUES],
            )
            + normalize_transform
        )

    def create_transformation(self) -> Transformation:
        """
        Add a transformation that replaces NaN in the input data with zeros,
        and mention whether the data was a NaN or not in another field.

        Returns:
        --------
        transformation: Transformation
            The chain of transformations defined for TACTiS.
        """
        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
            ]
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: nn.Module, device: torch.device
    ) -> PyTorchPredictor:
        """
        Create the predictor which can be used by GluonTS to do inference.

        Parameters:
        -----------
        transformation: Transformation
            The transformation to apply to the data prior to being sent to the model.
        trained_network: nn.Module
            An instance of PACTiSTrainingNetwork with trained parameters.
        device: torch.device
            The device where the model parameters should be placed.

        Returns:
        --------
        predictor: PyTorchPredictor
            The PyTorchTS predictor object.
        """
        prediction_network = PACTiSPredictionNetwork(
            num_series=self.num_series,
            model_parameters=self.model_parameters,
            prediction_length=self.prediction_length,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device=device)
        copy_parameters(trained_network, prediction_network)

        output_transform = cdf_to_gaussian_forward_transform if self.cdf_normalization else None
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            output_transform=output_transform,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            # freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )
