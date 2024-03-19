from torch.utils.data import Dataset
import numpy as np
import os
import pickle
# ----------------- har -----------------
from .UCIHAR import UCI_HAR_DATA



data_dict = {"ucihar": UCI_HAR_DATA,}


class data_set(Dataset):
    def __init__(self, args, dataset, flag,filterLabel=None):
        """
        args : a dict , In addition to the parameters for building the model, the parameters for reading the data are also in here
        dataset : It should be implmented dataset object, it contarins train_x, train_y, vali_x,vali_y,test_x,test_y
        flag : (str) "train","test","vali"
        """
        self.args = args
        self.flag = flag

        self.data_x = dataset.normalized_data_x
        self.data_y = dataset.data_y

        if filterLabel != None:
            # Filter the Y DataFrame to only include desired labels
            self.data_y  = self.data_y [self.data_y["activity_id"].isin(filterLabel)]

            # Use the index of the filtered Y to filter X
            self.data_x = self.data_x.loc[self.data_y.index]
            # drop some data
            self.data_x = self.data_x.sample(frac=0.8, random_state=1)
            self.data_y = self.data_y.loc[self.data_x.index]

        self.slidingwindows = dataset.slidingwindows
        self.act_weights = dataset.act_weights

        if self.args.model_type in ["freq", "cross"]:
            self.freq_path = dataset.freq_path
            self.freq_file_name = dataset.freq_file_name

        if self.flag == "train":
            # load train
            self.window_index = dataset.train_window_index
            print("Train data number : ", len(self.window_index))


        elif self.flag == "vali":
            # load vali

            self.window_index = dataset.vali_window_index
            print("Validation data number : ", len(self.window_index))


        else:
            # load test
            self.window_index = dataset.test_window_index
            print("Test data number : ", len(self.window_index))

        all_labels = list(np.unique(dataset.data_y))
        to_drop = list(dataset.drop_activities)
        label = [item for item in all_labels if item not in to_drop]
        self.nb_classes = len(label)
        assert self.nb_classes == len(dataset.no_drop_activites)

        classes = dataset.no_drop_activites
        self.class_transform = {x: i for i, x in enumerate(classes)}
        self.class_back_transform = {i: x for i, x in enumerate(classes)}
        self.input_length = self.slidingwindows[0][2] - self.slidingwindows[0][1]
        self.channel_in = self.data_x.shape[1] - 2

        if self.flag == "train":
            print("The number of classes is : ", self.nb_classes)
            print("The input_length  is : ", self.input_length)
            print("The channel_in is : ", self.channel_in)

    def __getitem__(self, index):
        # print(index)
        index = self.window_index[index]
        start_index = self.slidingwindows[index][1]
        end_index = self.slidingwindows[index][2]

        if self.args.model_type == "time":

            sample_x = self.data_x.iloc[start_index:end_index, 1:-1].values

            sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]

            return sample_x, sample_y, sample_y

        elif self.args.model_type == "freq":
            # sample_x = self.spec_list[index]
            with open(os.path.join(self.freq_path, "{}.pickle".format(self.freq_file_name[index])), 'rb') as handle:
                sample_x = pickle.load(handle)

            sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]
            return sample_x, sample_y, sample_y

        else:

            sample_ts_x = self.data_x.iloc[start_index:end_index, 1:-1].values
            # print(sample_ts_x.shape)

            with open(os.path.join(self.freq_path, "{}.pickle".format(self.freq_file_name[index])), 'rb') as handle:
                sample_fq_x = pickle.load(handle)
            # print(sample_fq_x.shape)
            sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]

            return sample_ts_x, sample_fq_x, sample_y

    def __len__(self):
        return len(self.window_index)
