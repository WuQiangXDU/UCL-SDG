import os
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np

#Digital data was collected at 12,000 samples per second
signal_size = 1024

# Four working conditions
dataname= {0:["97.mat","105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat","234.mat"],  # 1797rpm
           1:["98.mat","106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat","235.mat"],  # 1772rpm
           2:["99.mat","107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat","236.mat"],  # 1750rpm
           3:["100.mat","108.mat", "121.mat","133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat","237.mat"]}  # 1730rpm

datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal"]
axis = ["_DE_time", "_FE_time", "_BA_time"]

def get_files(root, N, signal_size, num_classes):

    # different domains with different labels
    label = []
    data = []
    lab = []
    if len(N) > 1:
        for i in N:
            label.append([j + (10 * i) for j in range(0, num_classes)])
    else:
        label = [i for i in range(0, num_classes)]
    for k in range(len(N)):
        label_new = label[k] if (len(N) > 1) else label
        for n in tqdm(range(len(dataname[N[k]]))):
            if n == 0:
                path1 = os.path.join(root, datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root, datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_load(path1, dataname[N[k]][n], label=label_new[n])
            data += data1
            lab += lab1

    return data, lab


def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    fl = fl.reshape(-1,)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        x = fl[start:end]
        x = np.abs(np.fft.fft(x))
        x = x / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1,1)
        x = (x - x.mean()) / x.std()

        data.append(x)
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

class CWRU(object):
    num_classes = 10
    inputchannel = 1
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.source_N = args.transfer_task[0]
        self.target_N = list(filter(lambda x: x != args.transfer_task[0][0], args.transfer_task[1]))  # Remove source domain code
        self.signal_size = args.signal_size
        self.num_classes = args.num_class

    def data_split(self,):
        train_data, train_label = get_files(self.data_dir, self.source_N, self.signal_size, self.num_classes)
        train_data = np.array(train_data)
        train_label = np.array(train_label)

        test_data, test_label = get_files(self.data_dir, self.target_N, self.signal_size, self.num_classes)
        test_data = np.array(test_data)
        test_label = np.array(test_label)
        return train_data, train_label, test_data, test_label, self.num_classes