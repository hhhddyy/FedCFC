from __future__ import annotations

from fedcfc.data import ACTIVITY_NAMES, SIGNALS, load_uci_har, make_uci_har_loaders
from fedcfc.federated import FedCFCServer
from fedcfc.training import build_training_parser, choose_device, prepare_training, train_clients


def parse_args():
    return build_training_parser().parse_args()


def main() -> None:
    args = parse_args()
    config, device, output_dir = prepare_training(args)
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
        "bptt_steps": args.bptt_steps,
    }
    preprocessing = {
        "mean": mean, "std": std, "signals": list(SIGNALS[:train_x.shape[-1]]),
        "timespan_seconds": 0.02, "sequence_length": train_x.shape[1],
    }
    train_clients(
        args, client_loaders, test_loader,
        client_labels=[loader.dataset.tensors[1] for loader in client_loaders],
        model_config=model_config, preprocessing=preprocessing,
        class_names=list(ACTIVITY_NAMES[:num_classes]), config=config, server=server,
        device=device, output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
