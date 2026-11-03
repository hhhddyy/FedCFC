from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class LeCunTanh(nn.Module):
    """Scaled hyperbolic tangent activation."""

    def forward(self, value: Tensor) -> Tensor:
        return 1.7159 * torch.tanh(0.666 * value)


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "gelu": nn.GELU,
        "lecun": LeCunTanh,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as error:
        choices = ", ".join(sorted(activations))
        raise ValueError(f"unknown activation {name!r}; choose one of {choices}") from error


class TwoLayerModule(nn.Module):
    """Two-layer module with an optional output activation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: str,
        *,
        activate_output: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.hidden_activation = _activation(activation)
        self.output_activation = _activation(activation) if activate_output else nn.Identity()

    def forward(self, value: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        hidden = self.hidden_activation(self.hidden(value))
        output = self.output_activation(self.output(hidden))
        return output, (hidden, output)


class FedCfCCell(nn.Module):
    """A CfC cell with explicit f, g, and h modules."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        backbone_units: int = 64,
        module_units: int = 64,
        activation: str = "lecun",
        bounded_f: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone = nn.Linear(input_size + hidden_size, backbone_units)
        self.backbone_activation = _activation(activation)
        self.bounded_f = bounded_f
        self.f = TwoLayerModule(
            backbone_units, module_units, hidden_size, activation,
            activate_output=bounded_f,
        )
        self.g = TwoLayerModule(backbone_units, module_units, hidden_size, activation)
        self.h = TwoLayerModule(backbone_units, module_units, hidden_size, activation)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim == 2:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)

    def forward(
        self, inputs: Tensor, hidden: Tensor, timespan: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if timespan.ndim == 1:
            timespan = timespan.unsqueeze(-1)
        features = torch.cat((inputs, hidden), dim=-1)
        features = self.backbone_activation(self.backbone(features))
        f_value, f_activations = self.f(features)
        g_value, _ = self.g(features)
        h_value, _ = self.h(features)
        gate = torch.sigmoid(-f_value * timespan)
        return gate * g_value + h_value, f_activations

    def f_parameters(self) -> Sequence[nn.Parameter]:
        """Return the parameters of f."""
        return tuple(self.f.parameters())

    def communicated_parameter(self) -> nn.Parameter:
        """Return the final f weight whose gradient is communicated."""
        return self.f.output.weight


class FedCFCClassifier(nn.Module):
    """Sequence classifier built from a FedCfCCell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        *,
        backbone_units: int = 64,
        module_units: int = 64,
        activation: str = "lecun",
        bounded_f: bool = True,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if pooling not in ("last", "mean"):
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling = pooling
        self.hidden_size = hidden_size
        self.cell = FedCfCCell(
            input_size,
            hidden_size,
            backbone_units=backbone_units,
            module_units=module_units,
            activation=activation,
            bounded_f=bounded_f,
        )
        self.readout = nn.Linear(hidden_size, num_classes)
        self._activity_score: Tensor | None = None

    def forward(self, inputs: Tensor, timespans: Tensor | None = None) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"inputs must have shape [batch, time, features], got {tuple(inputs.shape)}"
            )
        batch_size, sequence_length, _ = inputs.shape
        if sequence_length == 0:
            raise ValueError("inputs must contain at least one timestep")
        if timespans is None:
            timespans = inputs.new_ones((batch_size, sequence_length))
        if timespans.shape[:2] != inputs.shape[:2]:
            raise ValueError("timespans must have shape [batch, time] or [batch, time, 1]")

        hidden = inputs.new_zeros((batch_size, self.hidden_size))
        hidden_states: list[Tensor] = []
        layer_activations: list[list[Tensor]] = [[], []]
        for step in range(sequence_length):
            hidden, activations = self.cell(
                inputs[:, step], hidden, timespans[:, step]
            )
            if self.pooling == "mean":
                hidden_states.append(hidden)
            for layer, activation in enumerate(activations):
                layer_activations[layer].append(activation)

        # Use the minimum of the maximum absolute activations per layer.
        per_layer_maxima = [
            torch.stack(values, dim=1).abs().amax()
            for values in layer_activations
        ]
        self._activity_score = torch.stack(per_layer_maxima).amin()
        if self.pooling == "mean":
            hidden = torch.stack(hidden_states, dim=1).mean(dim=1)
        return self.readout(hidden)

    @property
    def activity_score(self) -> Tensor:
        if self._activity_score is None:
            raise RuntimeError("activity_score is available only after a forward pass")
        return self._activity_score

    def f_parameters(self) -> Sequence[nn.Parameter]:
        return self.cell.f_parameters()

    def communicated_parameter(self) -> nn.Parameter:
        return self.cell.communicated_parameter()

