

``` Markdown
# FedCFC: Federated Learning with CFC Networks

This repository contains the simulation code for the paper "FedCFC: On-Device Personalized Federated Learning with Closed-Form Continuous-Time Neural Networks". Our work focuses on incorporating Closed-form continuous-time (CFC) neural networks into federated learning, specifically designed for Internet of Things (IoT) devices with limited computational resources.

## Abstract
Closed-form continuous-time neural networks demonstrate exceptional expressivity in modeling time-series data and lower training and inference overheads, which make them suitable for microcontroller-based platforms. FedCFC proposes an innovative federated learning approach, efficiently managing non-IID data distributions among clients in a distributed IoT setting.


### Installation
A step-by-step series of examples that tell you how to get a development environment running. For example:

```sh
git clone https://github.com/your-repo/fedcfc.git
cd fedcfc
pip install -r requirements.txt
```

### Running the model

```sh
python main.py --num_clients 10 --rounds 20 --epochs 5 -- model CFC
```

## Directory Structure

- `/models`: Pre-trained CFC neural network models.
- `/venv`: Python virtual environment for managing dependencies.
- `/Fed`: Implementation of the federated learning algorithm.
- `/main.py`: Entry point for running simulation scripts.

```

```

