


# FedCFC: Federated Learning with CFC Networks

This repository contains the simulation code for the paper "FedCFC: On-Device Personalized Federated Learning with Closed-Form Continuous-Time Neural Networks". Our work focuses on incorporating Closed-form continuous-time (CFC) neural networks into federated learning, specifically designed for Internet of Things (IoT) devices with limited computational resources.

## Abstract
Closed-form continuous-time neural networks demonstrate exceptional expressivity in modeling time-series data and lower training and inference overheads, which make them suitable for microcontroller-based platforms. FedCFC proposes an innovative federated learning approach, efficiently managing non-IID data distributions among clients in a distributed IoT setting.



### Running the model

```sh
python main.py --data pathToDataset --modelType CFC
```

## Directory Structure

- `/models`:  CFC and LTC neural network models.
- `/Fed`: Implementation of the federated learning server and client class.
- `/main.py`: Entry point for running simulation scripts.
- `/data`: Training and Test Data

```

```

