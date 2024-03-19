import json
import queue
import sys
import threading
from random import random

from Fed.Server import Server
from Fed.client import Client
from models.CFC import CFC
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataPre import data_set,data_dict
import numpy as np
from utils import dotdict
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, help="Path to the dataset")
parser.add_argument("-savePath", type=str, default=".",help="Path to the saved model")
parser.add_argument("--norm",default="standardization",help="Normalization Method for input")
parser.add_argument("--inputTransform",default="time",help="if you want to use raw time series as input, set model_type as time ,"
                                                           "if you want to use spectrogram as input, set model_type as freq,"
                                                           "if you want to use both as inputs, set model_type as cross")
parser.add_argument("--wave",default="morl")
parser.add_argument("--expModel",default="Given")
parser.add_argument("--batchSize",default=32,type=int)
parser.add_argument("--shuffle",default=True,action="store_true")
parser.add_argument("--dropLast",default=False,action="store_true")
parser.add_argument("--datasetCfg",default="UCI",type=str,help="")
parser.add_argument("--modelType",default="CFC",type=str,help="")
parser.add_argument("--numClients",default=128,type=int,help="")
parser.add_argument("--fedRounds",default=200,type=int,help="")

argsCmd = parser.parse_args()
args = dotdict()


# Function for client training, which will be run in a separate thread
def client_training(client, barrier, update_queue):
    grad_vector, update_norm = client.get_gradient_vector()

    # Put the client's gradient and its norm into the queue
    update_queue.put((grad_vector, update_norm))

    # Wait for other clients to reach this point
    barrier.wait()

# ================ change the path to dataset !
args.root_path =  argsCmd.data
# ================ If using spectrogram as input, modify the path where spectrogram is stored
args.freq_save_path = argsCmd.savePath
# ================
if argsCmd.datasetCfg == "UCI":
    f = open('configs/UCI.json')
    cfg = json.load(f)
args.data_name = cfg.get("name")

args.difference = cfg.get("difference")
args.sampling_freq =   cfg.get("sp")
args.train_vali_quote = cfg.get("t_v_s")
args.windowsize = int(2.56 * args.sampling_freq)
args.drop_long = cfg.get("drop_long")

args.datanorm_type = argsCmd.norm

args.wavename = argsCmd.wave
# if you want to use raw time series as input, set model_type as time
# if you want to use spectrogram as input, set model_type as freq
# if you want to use both as inputs, set model_type as cross
args.model_type = argsCmd.inputTransform


# if you want to do the given train test experiment, set exp_mode as "Given"
# if you want to do Semi non overlapping experiment, set exp_mode as "SOCV"
# if you want to do Full non overlapping experiment, set exp_mode as "FOCV"
args.exp_mode = argsCmd.expModel

if args.exp_mode == "FOCV":
    args.displacement =  int(1 * args.windowsize)
else:
    args.displacement =  int(0.5 * args.windowsize)


args.batch_size = argsCmd.batchSize
args.shuffle = argsCmd.shuffle
args.drop_last = argsCmd.dropLast
dataset = data_dict[args.data_name](args)



# Initialize clients
num_clients = argsCmd.numClients

#todo add hyparms
hyparams = {"backbone_activation":"silu","no_gate":True,"backbone_layers":2,"backbone_units":64,"backbone_dr":0.1}



print("================ {} Mode ====================".format(dataset.exp_mode))
print("================ {} CV ======================".format(dataset.num_of_cv))
for i in range(dataset.num_of_cv):
    print("================ {Initialize Server}")
    # Initialize the server
    initial_vector = np.random.randn(cfg.get("hiddenSize"))  # Replace with the actual initial vector from the server
    server = Server(client_gradients=[initial_vector])

    clients = []
    for i in range(num_clients):
        dataset.update_train_val_test_keys()
        train_data = data_set(args, dataset, "train",filterLabel=[random.randint(1, cfg.get("maxLabel")+1)])
        test_data = data_set(args, dataset, "test")
        vali_data = data_set(args, dataset, "vali")

        # form the dataloader
        train_loader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  num_workers=0,
                                  drop_last=args.drop_last)

        vali_loader = DataLoader(vali_data,
                                 batch_size=args.batch_size,
                                 shuffle=args.shuffle,
                                 num_workers=0,
                                 drop_last=args.drop_last)

        test_loader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 shuffle=args.shuffle,
                                 num_workers=0,
                                 drop_last=args.drop_last)
        client = Client(i,CFC(cfg.get("inputSize"),cfg.get("hiddenSize",cfg.get("outF"),hyparams)), train_loader, 'NLL',initial_vector,1)
        clients.append(client)

        barrier = threading.Barrier(parties=num_clients)
        update_queue = queue.Queue()

        for round in range(argsCmd.fedRounds):
            # Start client training threads
            threads = []
            for client in clients:
                t = threading.Thread(target=client_training, args=(client, barrier, update_queue))
                threads.append(t)
                t.start()

            # Collect updates from clients
            client_gradients = []
            while any(t.is_alive() for t in threads):
                try:
                    grad_vector, update_norm = update_queue.get(timeout=None)
                    if update_norm < 0.1:
                        continue  # Ignore updates smaller than the threshold
                    client_gradients.append(grad_vector)
                except queue.Empty:
                    pass  # Timeout passed without receiving an update

            # Wait for all threads to complete
            for thread in threads:
                thread.join()



            # Server sends vector to clients
            for client in clients:
                client.d = server.phi.value

            # Clients train and send vector back to server
            client_gradients = []
            total_norm = 0
            for _ in range(num_clients):
                grad_vector, update_norm = update_queue.get()  # Blocking get, assuming all threads will put something
                client_gradients.append(grad_vector)
                total_norm += update_norm

            # Calculate average gradient norm
            average_norm = total_norm / num_clients

            for client in clients:
                client.m = average_norm

            # Server updates the vector
            server.client_gradients = client_gradients
            server.update_vector()

        for client_id, client in enumerate(clients):
            # Assuming you have a directory named 'saved_models'
            client.save_model(f'savedModels/{args.data_name}/client_model_{client_id}.pt')
