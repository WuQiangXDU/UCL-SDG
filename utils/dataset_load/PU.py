import os
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np

#Digital data was collected at 40,000 samples per second
signal_size = 1024

#1 Undamaged (healthy) bearings(6X)
HBdata = ['K001',"K002",'K003','K004','K005','K006']

#2 Artificially damaged bearings(12X)
ADBdata = ['KA01','KA03','KA05','KA06','KA07','KA08','KA09','KI01','KI03','KI05','KI07','KI08']

#3 Bearings with real damages caused by accelerated lifetime tests(14x)
RDBdata = ['KA04','KA16','KA30','KB23','KB24','KI21']

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
#state = WC[0] #WC[0] can be changed to different working states

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
        state = WC[N[k]]
        label_new = label[k] if (len(N) > 1) else label
        for n in tqdm(range(len(RDBdata))):
            for w3 in range(1):
                name3 = state + "_" + RDBdata[n] + "_" + str(w3 + 1)
                path3 = os.path.join(root, RDBdata[n], name3 + ".mat")
                data3, lab3 = data_load(path3, name=name3, label=label_new[n])
                data += data3
                lab += lab3

    return data, lab

def data_load(filename,name,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  #Take out the data

    fl = fl.reshape(-1,)

    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        x = fl[start:end]
        x = np.abs(np.fft.fft(x))
        x = x / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1,1)
        data.append(x)
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data, lab

class PU(object):
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


