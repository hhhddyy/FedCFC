from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import FedCFCClassifier


@dataclass(frozen=True)
class FBindConfig:
    """Local optimization and regularization settings."""

    learning_rate: float = 0.01
    lambda_direction: float = 0.6
    lambda_magnitude: float = 0.2
    lambda_activity: float = 0.2
    activity_threshold: float = 1.59
    epsilon: float = 1e-8
    gradient_clip_norm: float | None = None
    optimizer: str = "sgd"

    def __post_init__(self) -> None:
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer must be 'sgd' or 'adam'")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


@dataclass(frozen=True)
class CommonUpdate:
    """Common update direction (CUD) and magnitude (CUM)."""

    direction: Tensor
    magnitude: Tensor

    def to(self, device: torch.device | str) -> "CommonUpdate":
        return CommonUpdate(self.direction.to(device), self.magnitude.to(device))


@dataclass(frozen=True)
class ClientStepResult:
    task_loss: float
    total_loss: float
    directional_penalty: float
    magnitude_penalty: float
    activity_penalty: float
    activity_score: float
    accuracy: float
    uploaded_gradient: Tensor | None


class FedCFCClient:
    """A personalized client model and its bound local updater."""

    def __init__(
        self,
        model: FedCFCClassifier,
        config: FBindConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config or FBindConfig()
        self.device = torch.device(device)
        optimizer_class = torch.optim.Adam if self.config.optimizer == "adam" else torch.optim.SGD
        self.optimizer = optimizer_class(
            self.model.parameters(), lr=self.config.learning_rate
        )

    def _check_gradients(self) -> None:
        finite = [torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()
                  if parameter.grad is not None]
        if finite and not torch.stack(finite).all():
            raise FloatingPointError("client produced non-finite gradients before the optimizer update")

    def train_batch(
        self,
        inputs: Tensor,
        targets: Tensor,
        timespans: Tensor | None = None,
        common_update: CommonUpdate | None = None,
        *,
        local_only: bool = False,
    ) -> ClientStepResult:
        """Run fBind, or task-only local training without a gradient upload."""
        if local_only and common_update is not None:
            raise ValueError("local-only training cannot receive a common update")
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        if timespans is not None:
            timespans = timespans.to(self.device)
        if common_update is not None:
            common_update = common_update.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(inputs, timespans)
        task_loss = F.cross_entropy(logits, targets)
        if local_only:
            if not torch.isfinite(task_loss):
                raise FloatingPointError("client produced a non-finite task loss")
            task_loss.backward()
            if self.config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self._check_gradients()
            self.optimizer.step()
            return ClientStepResult(
                task_loss=float(task_loss.detach()),
                total_loss=float(task_loss.detach()),
                directional_penalty=0.0,
                magnitude_penalty=0.0,
                activity_penalty=0.0,
                activity_score=float(self.model.activity_score.detach()),
                accuracy=float((logits.argmax(dim=1) == targets).float().mean()),
                uploaded_gradient=None,
            )
        communicated_parameter = self.model.communicated_parameter()

        has_bound = common_update is not None
        if common_update is not None:
            if common_update.direction.numel() != communicated_parameter.numel():
                raise ValueError("CUD must match the communicated f weight size")
            if not torch.isfinite(common_update.direction).all():
                raise ValueError("CUD must be finite")
            if (
                common_update.magnitude.ndim != 0
                or not torch.isfinite(common_update.magnitude)
                or common_update.magnitude < 0
            ):
                raise ValueError("CUM must be a finite non-negative scalar")
        task_gradient = torch.autograd.grad(
            task_loss,
            communicated_parameter,
            create_graph=has_bound,
            retain_graph=True,
        )[0]
        flat_task_gradient = task_gradient.reshape(-1)

        zero = task_loss.new_zeros(())
        if has_bound:
            assert common_update is not None
            direction = common_update.direction.reshape(-1)
            if direction.norm() > self.config.epsilon:
                directional_penalty = (
                    1.0
                    - F.cosine_similarity(
                        flat_task_gradient.unsqueeze(0),
                        direction.unsqueeze(0),
                        dim=1,
                        eps=self.config.epsilon,
                    ).squeeze(0)
                ).clamp(min=0.0, max=2.0)
            else:
                # Retain magnitude regularization when the direction is undefined.
                directional_penalty = zero
            magnitude_penalty = F.relu(
                torch.linalg.vector_norm(flat_task_gradient)
                - common_update.magnitude
            )
        else:
            directional_penalty = zero
            magnitude_penalty = zero

        # Penalize activity below the threshold.
        activity_penalty = F.relu(
            self.config.activity_threshold - self.model.activity_score
        )
        total_loss = (
            task_loss
            + self.config.lambda_direction * directional_penalty
            + self.config.lambda_magnitude * magnitude_penalty
            + self.config.lambda_activity * activity_penalty
        )

        f_parameters = tuple(self.model.f_parameters())
        f_parameter_ids = {id(parameter) for parameter in f_parameters}
        personalized_parameters = tuple(
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in f_parameter_ids
        )

        # Regularize f; use task gradients for all other parameters.
        personalized_gradients = torch.autograd.grad(
            task_loss,
            personalized_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        f_gradients = torch.autograd.grad(
            total_loss,
            f_parameters,
            allow_unused=True,
        )
        for parameter, gradient in zip(
            personalized_parameters, personalized_gradients, strict=True
        ):
            parameter.grad = None if gradient is None else gradient.detach()
        for parameter, gradient in zip(f_parameters, f_gradients, strict=True):
            parameter.grad = None if gradient is None else gradient.detach()

        if self.config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
        if communicated_parameter.grad is None:
            raise RuntimeError("the communicated f parameter did not receive a gradient")
        uploaded_gradient = communicated_parameter.grad.detach().clone().cpu()
        if not torch.isfinite(uploaded_gradient).all():
            raise FloatingPointError("client produced a non-finite local update")
        self._check_gradients()
        self.optimizer.step()

        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        return ClientStepResult(
            task_loss=float(task_loss.detach()),
            total_loss=float(total_loss.detach()),
            directional_penalty=float(directional_penalty.detach()),
            magnitude_penalty=float(magnitude_penalty.detach()),
            activity_penalty=float(activity_penalty.detach()),
            activity_score=float(self.model.activity_score.detach()),
            accuracy=float(accuracy.detach()),
            uploaded_gradient=uploaded_gradient,
        )


class FedCFCServer:
    """fBind server with closed-form or numerical SDP CUD aggregation."""

    def __init__(
        self,
        *,
        epsilon: float = 1e-8,
        cud_method: str = "closed",
        solver_tolerance: float = 1e-6,
        solver_max_iterations: int = 20000,
    ) -> None:
        if cud_method not in ("closed", "solver"):
            raise ValueError("cud_method must be 'closed' or 'solver'")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(solver_tolerance) or solver_tolerance <= 0:
            raise ValueError("solver_tolerance must be finite and positive")
        if solver_max_iterations < 1:
            raise ValueError("solver_max_iterations must be positive")
        if cud_method == "solver":
            from .cud import require_solver

            require_solver()
        self.epsilon = epsilon
        self.cud_method = cud_method
        self.solver_tolerance = solver_tolerance
        self.solver_max_iterations = solver_max_iterations
        self.solver_diagnostics: dict[str, str | int | float] | None = None
        self.common_update: CommonUpdate | None = None

    def aggregate(self, local_updates: Iterable[Tensor]) -> CommonUpdate:
        """Compute the common direction and mean norm of client gradients."""
        updates = tuple(update.detach().float().cpu() for update in local_updates)
        if not updates:
            raise ValueError("at least one local update is required")
        expected_shape = updates[0].shape
        if updates[0].numel() == 0:
            raise ValueError("local updates must not be empty")
        if any(update.shape != expected_shape for update in updates):
            raise ValueError("all local updates must have the same shape")
        stacked = torch.stack([update.reshape(-1) for update in updates])
        if not torch.isfinite(stacked).all():
            raise FloatingPointError("cannot aggregate non-finite local updates")

        norms = torch.linalg.vector_norm(stacked, dim=1)
        magnitude = norms.mean()
        valid = norms > self.epsilon
        diagnostics = None
        if valid.any():
            unit_updates = stacked[valid] / norms[valid].unsqueeze(1)
            resultant = unit_updates.sum(dim=0)
            resultant_norm = torch.linalg.vector_norm(resultant)
            if resultant_norm > self.epsilon:
                if self.cud_method == "solver":
                    from .cud import solve_cud_sdp

                    direction, diagnostics = solve_cud_sdp(
                        unit_updates,
                        tolerance=self.solver_tolerance,
                        max_iterations=self.solver_max_iterations,
                    )
                else:
                    direction = resultant / resultant_norm
            else:
                # Return zero when client directions cancel.
                direction = torch.zeros_like(resultant)
                diagnostics = {"status": "zero_resultant"}
        else:
            direction = torch.zeros_like(stacked[0])
            diagnostics = {"status": "zero_gradients"}
        self.solver_diagnostics = diagnostics if self.cud_method == "solver" else None
        self.common_update = CommonUpdate(direction, magnitude)
        return self.common_update


@torch.no_grad()
def evaluate_client(
    client: FedCFCClient,
    batches: Iterable[tuple[Tensor, Tensor, Tensor]],
) -> tuple[float, float]:
    """Return average loss and accuracy for one personalized client."""
    client.model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets, timespans in batches:
        inputs = inputs.to(client.device)
        targets = targets.to(client.device)
        timespans = timespans.to(client.device)
        logits = client.model(inputs, timespans)
        total_loss += float(F.cross_entropy(logits, targets, reduction="sum"))
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total_examples += targets.numel()
    if total_examples == 0:
        raise ValueError("evaluation loader is empty")
    return total_loss / total_examples, total_correct / total_examples

