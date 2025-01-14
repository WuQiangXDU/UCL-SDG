import os
from tqdm import tqdm
import numpy as np
signal_size = 1024

# Three working conditions
dataname= {0:["ib600_2.csv","n600_3_2.csv","ob600_2.csv","tb600_2.csv"],
           1:["ib800_2.csv","n800_3_2.csv","ob800_2.csv","tb800_2.csv"],
           2:["ib1000_2.csv","n1000_3_2.csv","ob1000_2.csv","tb1000_2.csv"]}

#generate Training Dataset and Testing Dataset
def get_files(root, N, num_classes):

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
            path1 = os.path.join(root, dataname[N[k]][n])
            data1, lab1 = data_load(path1, label=label_new[n])
            data += data1
            lab += lab1

    return data, lab

def data_load(filename,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = np.loadtxt(filename)
    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
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


class JNU(object):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.source_N = args.transfer_task[0]
        self.target_N = list(
            filter(lambda x: x != args.transfer_task[0][0], args.transfer_task[1]))  # Remove source domain code
        self.num_classes = args.num_class

    def data_split(self, ):
        train_data, train_label = get_files(self.data_dir, self.source_N, self.num_classes)
        train_data = np.array(train_data)
        train_label = np.array(train_label)

        test_data, test_label = get_files(self.data_dir, self.target_N, self.num_classes)
        test_data = np.array(test_data)
        test_label = np.array(test_label)
        return train_data, train_label, test_data, test_label, self.num_classes



