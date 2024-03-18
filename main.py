import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataPre import data_set,data_dict
import numpy as np
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = dotdict()

# ================ change the path to dataset !
args.root_path =  r"C:\Users\hhhddyy\Desktop\NTU-Tan\phd\FedCFC\data\UCIHAR"
# ================ If using spectrogram as input, modify the path where spectrogram is stored
args.freq_save_path = None
# ================
args.data_name = "ucihar"

args.difference = True
args.sampling_freq =   50
args.train_vali_quote = 0.95
# for this dataset the window size is 128
args.windowsize = int(2.56 * args.sampling_freq)

args.drop_long = False

args.datanorm_type = "standardization" # None ,"standardization", "minmax"

args.wavename = "morl"
# if you want to use raw time series as input, set model_type as time
# if you want to use spectrogram as input, set model_type as freq
# if you want to use both as inputs, set model_type as cross
args.model_type = "time"


# if you want to do the given train test experiment, set exp_mode as "Given"
# if you want to do Semi non overlapping experiment, set exp_mode as "SOCV"
# if you want to do Full non overlapping experiment, set exp_mode as "FOCV"
args.exp_mode = "Given"

if args.exp_mode == "FOCV":
    args.displacement =  int(1 * args.windowsize)
else:
    args.displacement =  int(0.5 * args.windowsize)


args.batch_size = 32
args.shuffle = True
args.drop_last = False
dataset = data_dict[args.data_name](args)

print("================ {} Mode ====================".format(dataset.exp_mode))
print("================ {} CV ======================".format(dataset.num_of_cv))
for i in range(dataset.num_of_cv):
    dataset.update_train_val_test_keys()
    train_data = data_set(args, dataset, "train")
    test_data = data_set(args, dataset, "test")
    vali_data = data_set(args, dataset, "vali")

    # form the dataloader
    train_data_loader = DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=args.shuffle,
                                   num_workers=0,
                                   drop_last=args.drop_last)

    vali_data_loader = DataLoader(vali_data,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  num_workers=0,
                                  drop_last=args.drop_last)

    test_data_loader = DataLoader(test_data,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  num_workers=0,
                                  drop_last=args.drop_last)