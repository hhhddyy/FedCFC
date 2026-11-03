from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def require_solver():
    """Validate and import the optional solver dependencies."""
    try:
        import cvxpy as cp
    except ImportError as error:
        raise RuntimeError(
            "CUD method 'solver' requires CVXPY/SCS; install with "
            "python -m pip install -e '.[solver]'"
        ) from error
    if "SCS" not in cp.installed_solvers():
        raise RuntimeError("CUD method 'solver' requires SCS; install the solver extra")
    return cp


def solve_cud_sdp(
    unit_updates: Tensor, *, tolerance: float, max_iterations: int
) -> tuple[Tensor, dict[str, str | int | float]]:
    """Solve the cosine-Gram SDP and recover a unit direction in gradient space."""
    cp = require_solver()
    units = unit_updates.detach().cpu().double().numpy()
    units = units / np.linalg.norm(units, axis=1, keepdims=True)
    count = len(units)
    gram = units @ units.T
    matrix = cp.Variable((count + 1, count + 1), symmetric=True)
    centroid_weights = np.full(count + 1, 1.0 / count)
    centroid_weights[-1] = -1.0
    objective_matrix = np.outer(centroid_weights, centroid_weights)
    problem = cp.Problem(
        cp.Minimize(cp.trace(objective_matrix @ matrix)),
        [matrix >> 0, matrix[:count, :count] == gram, matrix[count, count] == 1],
    )
    try:
        problem.solve(
            solver=cp.SCS, eps=tolerance, max_iters=max_iterations, verbose=False
        )
    except cp.error.SolverError as error:
        raise RuntimeError("SCS failed to solve the CUD SDP") from error
    if problem.status != cp.OPTIMAL or matrix.value is None:
        raise RuntimeError(
            f"CUD SDP did not converge: {problem.status}; increase "
            "--cud-solver-max-iters or adjust --cud-solver-tol"
        )

    solution = np.asarray(matrix.value)
    if not np.isfinite(solution).all():
        raise RuntimeError("CUD SDP returned non-finite values")
    solution = (solution + solution.T) / 2
    residual = max(
        float(np.max(np.abs(solution[:count, :count] - gram))),
        float(abs(solution[count, count] - 1)),
        float(max(0, -np.linalg.eigvalsh(solution)[0])),
    )
    check_tolerance = max(1e-5, 50 * tolerance)
    if residual > check_tolerance:
        raise RuntimeError(f"CUD SDP feasibility check failed: residual={residual:.3g}")

    cross_similarities = solution[:count, count]
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    # Exclude numerical null modes during inversion.
    retained = eigenvalues > max(1.0, float(eigenvalues[-1])) * 1e-10
    basis = eigenvectors[:, retained]
    coefficients = basis @ ((basis.T @ cross_similarities) / eigenvalues[retained])
    direction = units.T @ coefficients
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("CUD SDP could not recover a nonzero gradient direction")
    direction /= norm
    recovery_residual = float(np.max(np.abs(units @ direction - cross_similarities)))
    # Check the objective against the Cauchy-Schwarz bound.
    score = float((units @ direction).sum())
    objective_gap = max(0.0, float(np.linalg.norm(units.sum(axis=0))) - score)
    if recovery_residual > check_tolerance or objective_gap > count * check_tolerance:
        raise RuntimeError(
            "CUD SDP recovery check failed: "
            f"residual={recovery_residual:.3g}, objective_gap={objective_gap:.3g}"
        )
    diagnostics = {
        "status": str(problem.status),
        "iterations": int(problem.solver_stats.num_iters),
        "feasibility_residual": residual,
        "recovery_residual": recovery_residual,
        "cosine_objective_gap": objective_gap,
    }
    return torch.from_numpy(direction.copy()).to(dtype=unit_updates.dtype), diagnostics
