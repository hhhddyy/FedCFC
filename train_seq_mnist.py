from __future__ import annotations

from fedcfc.federated import FedCFCServer
from fedcfc.mnist import CLASS_NAMES, load_mnist, make_mnist_loaders
from fedcfc.training import build_training_parser, prepare_training, train_clients


def parse_args():
    parser = build_training_parser()
    parser.add_argument("--sequence-mode", choices=("rows", "pixels"), default="rows",
                        help="rows: 28 steps with 28 features; pixels: 784 steps with one feature")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, device, output_dir = prepare_training(args)
    server = None
    if args.training_mode == "federated":
        server = FedCFCServer(
            epsilon=config.epsilon, cud_method=args.cud_method,
            solver_tolerance=args.cud_solver_tol, solver_max_iterations=args.cud_solver_max_iters,
        )
    train = load_mnist(args.data_dir, train=True, download=args.download, sequence_mode=args.sequence_mode)
    test = load_mnist(args.data_dir, train=False, download=args.download, sequence_mode=args.sequence_mode)
    clients, test_loader, client_labels = make_mnist_loaders(
        train, test, num_clients=args.clients, alpha=args.dirichlet_alpha,
        batch_size=args.batch_size, seed=args.seed, partition=args.partition,
    )
    model_config = {
        "input_size": train.preprocessing["input_size"], "hidden_size": args.hidden_size,
        "num_classes": len(CLASS_NAMES), "backbone_units": args.backbone_units,
        "module_units": args.module_units, "activation": "lecun",
        "bounded_f": args.bounded == "yes", "pooling": args.pooling, "bptt_steps": args.bptt_steps,
    }
    train_clients(
        args, clients, test_loader, client_labels=client_labels, model_config=model_config,
        preprocessing=train.preprocessing, class_names=CLASS_NAMES, config=config, server=server,
        device=device, output_dir=output_dir,
        metadata={"dataset": "mnist", "sequence_mode": args.sequence_mode,
                  "sequence_length": train.preprocessing["sequence_length"], "timespan": 1.0},
    )


if __name__ == "__main__":
    main()
