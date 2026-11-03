from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections.abc import Iterator
from pathlib import Path
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from fedcfc.checkpoint import create_checkpoint, save_checkpoint
from fedcfc.data import ACTIVITY_NAMES, SIGNALS, load_uci_har, make_uci_har_loaders
from fedcfc.federated import (
    FBindConfig,
    FedCFCClient,
    FedCFCServer,
    evaluate_client,
)
from fedcfc.model import FedCFCClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--steps-per-round", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--partition", choices=("iid", "dirichlet"), default="dirichlet",
        help="stratified IID or Dirichlet class skew (default: dirichlet)",
    )
    parser.add_argument(
        "--dirichlet-alpha", type=float, default=0.3,
        help="class-skew concentration; used only with --partition dirichlet",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--backbone-units", type=int, default=64)
    parser.add_argument("--module-units", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="none",
                        help="constant learning rate or cosine decay stepped after each round")
    parser.add_argument("--min-learning-rate", type=float, default=0.0,
                        help="cosine decay floor; used only with --lr-scheduler cosine")
    parser.add_argument("--pooling", choices=("last", "mean"), default="last",
                        help="classify the final hidden state or the mean over all timesteps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--bounded", choices=("yes", "no"), default="yes",
        help="apply the final LeCun tanh to f (yes), or leave f linear (no)",
    )
    parser.add_argument(
        "--training-mode", choices=("federated", "local"), default="federated",
        help="fBind or independent task-only clients; local with --clients 1 uses all training data",
    )
    parser.add_argument(
        "--cud-method", choices=("closed", "solver"), default="closed",
        help="CUD aggregation: closed form or CVXPY/SCS cosine-Gram SDP",
    )
    parser.add_argument("--cud-solver-tol", type=float, default=1e-6)
    parser.add_argument("--cud-solver-max-iters", type=int, default=20000)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, or mps (default: auto)",
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--output-dir", help="checkpoint directory (default: a unique directory under runs/)")
    parser.add_argument("--save-every", type=int, default=5,
                        help="save periodic checkpoints every N rounds; always save the final round")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def next_batch(
    loader: DataLoader,
    iterator: Iterator | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], Iterator]:
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def main() -> None:
    args = parse_args()
    if min(args.rounds, args.steps_per_round, args.eval_every, args.save_every) < 1:
        raise ValueError("rounds, steps-per-round, eval-every, and save-every must be positive")
    config = FBindConfig(learning_rate=args.learning_rate, optimizer=args.optimizer)
    if args.lr_scheduler == "cosine" and (
        not math.isfinite(args.min_learning_rate)
        or not 0 <= args.min_learning_rate <= args.learning_rate
    ):
        raise ValueError("min-learning-rate must be finite and between zero and learning-rate")
    output_dir = Path(args.output_dir or f"runs/{uuid.uuid4().hex[:12]}").expanduser().resolve()
    if (output_dir / "latest.pt").exists() or (output_dir / "final.pt").exists() or list(output_dir.glob("round-*.pt")):
        raise FileExistsError(f"checkpoint directory already contains a run: {output_dir}; choose a new --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    # Validate solver dependencies before training.
    server = None
    if args.training_mode == "federated":
        server = FedCFCServer(
            epsilon=config.epsilon,
            cud_method=args.cud_method,
            solver_tolerance=args.cud_solver_tol,
            solver_max_iterations=args.cud_solver_max_iters,
        )

    train_x, train_y, test_x, test_y, mean, std = load_uci_har(
        args.data_dir, download=args.download, return_normalization=True
    )
    client_loaders, test_loader = make_uci_har_loaders(
        train_x,
        train_y,
        test_x,
        test_y,
        num_clients=args.clients,
        alpha=args.dirichlet_alpha,
        batch_size=args.batch_size,
        seed=args.seed,
        partition=args.partition,
    )
    num_classes = int(train_y.max()) + 1
    model_config = {
        "input_size": train_x.shape[-1], "hidden_size": args.hidden_size,
        "num_classes": num_classes, "backbone_units": args.backbone_units,
        "module_units": args.module_units, "activation": "lecun",
        "bounded_f": args.bounded == "yes",
        "pooling": args.pooling,
    }
    initial_model = FedCFCClassifier(**model_config)
    preprocessing = {
        "mean": mean, "std": std, "signals": list(SIGNALS[:train_x.shape[-1]]),
        "timespan_seconds": 0.02, "sequence_length": train_x.shape[1],
    }
    clients = [
        FedCFCClient(copy.deepcopy(initial_model), config, device=device)
        for _ in range(args.clients)
    ]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            client.optimizer, T_max=args.rounds, eta_min=args.min_learning_rate
        )
        for client in clients
    ] if args.lr_scheduler == "cosine" else []
    iterators: list[Iterator | None] = [None] * args.clients
    client_samples = [len(loader.dataset) for loader in client_loaders]
    client_class_counts = [
        torch.bincount(loader.dataset.tensors[1], minlength=num_classes).tolist()
        for loader in client_loaders
    ]

    print(
        json.dumps(
            {
                "device": str(device),
                "output_dir": str(output_dir),
                "save_every": args.save_every,
                "training_mode": args.training_mode,
                "bounded": args.bounded,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "lr_scheduler": args.lr_scheduler,
                "min_learning_rate": args.min_learning_rate if schedulers else None,
                "pooling": args.pooling,
                "cud_method": args.cud_method if server is not None else None,
                "clients": args.clients,
                "train_samples": len(train_y),
                "test_samples": len(test_y),
                "seed": args.seed,
                "batch_size": args.batch_size,
                "steps_per_round": args.steps_per_round,
                "partition": args.partition,
                "dirichlet_alpha": args.dirichlet_alpha if args.partition == "dirichlet" else None,
                "client_samples": client_samples,
                "client_class_counts": client_class_counts,
            }
        )
    )
    for round_index in range(1, args.rounds + 1):
        learning_rate = clients[0].optimizer.param_groups[0]["lr"]
        uploads = []
        task_losses = []
        train_accuracies = []
        for client_index, (client, loader) in enumerate(zip(clients, client_loaders)):
            step_uploads = []
            for _ in range(args.steps_per_round):
                batch, iterators[client_index] = next_batch(
                    loader, iterators[client_index]
                )
                result = client.train_batch(
                    *batch,
                    common_update=server.common_update if server is not None else None,
                    local_only=server is None,
                )
                if server is not None:
                    assert result.uploaded_gradient is not None
                    step_uploads.append(result.uploaded_gradient)
                task_losses.append(result.task_loss)
                train_accuracies.append(result.accuracy)
            if server is not None:
                uploads.append(torch.stack(step_uploads).mean(dim=0))

        metrics = {
            "round": round_index,
            "learning_rate": learning_rate,
            "train_loss": float(np.mean(task_losses)),
            "train_accuracy": float(np.mean(train_accuracies)),
        }
        for scheduler in schedulers:
            scheduler.step()
        metrics["next_learning_rate"] = clients[0].optimizer.param_groups[0]["lr"]
        if server is not None:
            common_update = server.aggregate(uploads)
            metrics["cum"] = float(common_update.magnitude)
            if server.solver_diagnostics is not None:
                metrics["cud_solver"] = server.solver_diagnostics
        if round_index == 1 or round_index % args.eval_every == 0 or round_index == args.rounds:
            evaluations = [evaluate_client(client, test_loader) for client in clients]
            metrics["test_loss"] = float(np.mean([item[0] for item in evaluations]))
            metrics["test_accuracy"] = float(
                np.mean([item[1] for item in evaluations])
            )
        if round_index % args.save_every == 0 or round_index == args.rounds:
            checkpoint = create_checkpoint(
                [client.model for client in clients], model_config=model_config,
                preprocessing=preprocessing, class_names=list(ACTIVITY_NAMES[:num_classes]),
                run_config=vars(args), round_index=round_index, metrics=metrics,
                client_samples=client_samples, client_class_counts=client_class_counts,
            )
            if round_index % args.save_every == 0:
                save_checkpoint(checkpoint, output_dir / f"round-{round_index:06d}.pt")
            save_checkpoint(checkpoint, output_dir / "latest.pt")
            if round_index == args.rounds:
                save_checkpoint(checkpoint, output_dir / "final.pt")
            metrics["checkpoint"] = str(output_dir / ("final.pt" if round_index == args.rounds else "latest.pt"))
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()

