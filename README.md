**FedCFC: On-Device Personalized Federated Learning with Closed-Form Continuous-Time Neural Networks**

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install -e '.[solver]'
```

Train federated models with IID data:

```bash
CUDA_VISIBLE_DEVICES=0 python train_uci_har.py \
  --download --device cuda --training-mode federated \
  --partition iid --bounded no --cud-method closed \
  --clients 6 --rounds 100 --steps-per-round 5 \
  --output-dir runs/federated-iid --save-every 5
```

Train one model on all training data:

```bash
CUDA_VISIBLE_DEVICES=0 python train_uci_har.py \
  --download --device cuda --training-mode local --bounded no \
  --partition iid --clients 1 --batch-size 32 \
  --optimizer adam --learning-rate 0.001 \
  --lr-scheduler cosine --min-learning-rate 0.00001 --pooling mean \
  --rounds 30 --steps-per-round 230 --eval-every 1 \
  --output-dir runs/single --save-every 5
```

For one client with batch size 32, 230 steps cover one UCI HAR training epoch.
Cosine decay updates after each round and reaches its floor after the final round.
Logs show the learning rate used in each round and the next learning rate.
Use a new output directory for each run. Checkpoints include all client models:
`latest.pt`, `final.pt`, and periodic `round-XXXXXX.pt` files. They support
evaluation; training resumption is not implemented.

Training parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | `data` | Dataset directory. |
| `--download` | Off | Download and extract UCI HAR if needed. |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:N`, or `mps`; auto selects an available accelerator. |
| `--training-mode` | `federated` | `federated`: fBind; `local`: independent task-only training. |
| `--clients` | `6` | Number of models and training partitions; one client receives all training data. |
| `--partition` | `dirichlet` | `iid`: stratified balanced clients; `dirichlet`: class-skewed clients. |
| `--dirichlet-alpha` | `0.3` | Smaller values increase class skew; ignored for IID. |
| `--bounded` | `yes` | `yes`: final tanh on f; `no`: linear f output. Both retain the sigmoid gate. |
| `--cud-method` | `closed` | `closed`: closed-form CUD; `solver`: CVXPY/SCS SDP. Ignored in local mode. |
| `--cud-solver-tol` | `1e-6` | Numerical solver tolerance. |
| `--cud-solver-max-iters` | `20000` | Maximum solver iterations. |
| `--rounds` | `50` | Number of training rounds; a round is not generally an epoch. |
| `--steps-per-round` | `1` | Minibatch updates per client per round. |
| `--batch-size` | `32` | Training samples per minibatch. |
| `--hidden-size` | `64` | Recurrent state size. |
| `--backbone-units` | `64` | Backbone width. |
| `--module-units` | `64` | Hidden width of each f/g/h module. |
| `--optimizer` | `sgd` | `sgd` or `adam`, in local and federated modes. Adam uses betas `(0.9, 0.999)`, epsilon `1e-8`, and no weight decay. |
| `--learning-rate` | `0.01` | Initial learning rate; use `0.001` as a starting point for Adam. |
| `--lr-scheduler` | `none` | `none`: constant rate; `cosine`: decay over the configured rounds. |
| `--min-learning-rate` | `0.0` | Cosine decay floor; ignored when scheduling is disabled. |
| `--pooling` | `last` | Classify the final hidden state (`last`) or the average hidden state across timesteps (`mean`). |
| `--bptt-steps` | `0` | Stop recurrent gradients every N timesteps; `0` uses full backpropagation. Forward state persists across boundaries. |
| `--seed` | `7` | Random seed for initialization, partitioning, and sampling. |
| `--eval-every` | `5` | Evaluation interval in rounds; first and final rounds are also evaluated. |
| `--save-every` | `5` | Checkpoint interval in rounds; the final round is always saved. |
| `--output-dir` | Unique path under `runs/` | Directory for saved checkpoints. |

Evaluate a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_uci_har.py \
  --checkpoint runs/single/final.pt --device cuda \
  --output runs/single/test-metrics.json
```

Evaluation parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | Required | Saved checkpoint to evaluate. |
| `--client-id` | All clients | Evaluate one client by its zero-based ID. |
| `--batch-size` | `64` | Evaluation samples per minibatch. |
| `--output` | None | Also save the printed JSON report to this path. |

Evaluation also accepts `--data-dir`, `--download`, and `--device` as defined above.
The report includes per-client accuracy, loss, class recall, and confusion
matrices, plus mean accuracy and loss. Model settings and normalization are
loaded from the checkpoint.

Install MNIST support for the CUDA environment above:

```bash
pip install torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
```

Train one model on sequential MNIST:

```bash
python train_seq_mnist.py \
  --download --device cuda:0 --training-mode local \
  --clients 1 --partition iid --sequence-mode rows --bounded no \
  --optimizer adam --learning-rate 0.001 --lr-scheduler cosine \
  --min-learning-rate 0.00001 --pooling mean --batch-size 64 \
  --rounds 30 --steps-per-round 938 --eval-every 1 \
  --output-dir runs/mnist-local --save-every 5
```

Train federated MNIST models:

```bash
python train_seq_mnist.py \
  --download --device cuda:0 --training-mode federated \
  --clients 6 --partition dirichlet --dirichlet-alpha 0.3 \
  --sequence-mode rows --bounded no --optimizer adam --learning-rate 0.001 \
  --lr-scheduler cosine --min-learning-rate 0.00001 --pooling mean \
  --rounds 100 --steps-per-round 5 --output-dir runs/mnist-federated
```

MNIST accepts the training parameters above and adds:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--sequence-mode` | `rows` | `rows`: 28 timesteps with 28 features; `pixels`: the standard 784-timestep pixel sequence with one feature. Both scan left-to-right, top-to-bottom, without permutation. |

Pixels are divided by 255. The timestep is 1 sequence unit, not a physical time.
The official 60,000 training images are partitioned among clients; all clients
are evaluated on the official 10,000 test images. For one client and batch size
64, 938 steps cover one epoch. Pixel sequences take substantially longer than rows.

For pixel-wise training, replace `--sequence-mode rows` with
`--sequence-mode pixels --bptt-steps 32`. Full 784-step backpropagation produced
non-finite gradients in CUDA checks. Truncation limits gradient propagation
across timesteps; it does not reset the hidden state or shorten the input sequence.

Evaluate a MNIST checkpoint:

```bash
python evaluate_seq_mnist.py \
  --checkpoint runs/mnist-local/final.pt --device cuda:0 \
  --output runs/mnist-local/test-metrics.json
```

The evaluation options match those above. Sequence format and preprocessing
come from the checkpoint; no sequence-format argument is needed for evaluation.
