import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from utils.accuracy import accuracy

class dataset(Dataset):

    def __init__(self, list_data):
        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        label = self.labels[item]
        seq = np.array(seq)
        seq = np.float32(seq)
        seq = np.expand_dims(seq, axis=0)
        return seq, label


def model_test(model, test_loader):
    global futures
    first = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            new_repr = model(batch)
            if first == 0:
                futures = new_repr
                first = 1
                continue
            futures = torch.cat((futures, new_repr), dim=0)
            torch.cuda.empty_cache()
    return futures


def eval_classification(model, all_data_label, args, device):

    train_data = all_data_label[0]
    train_labels = all_data_label[1]
    test_data = all_data_label[2]
    test_labels = all_data_label[3]

    train_data = torch.tensor(train_data).transpose(1, 2).type(torch.FloatTensor).to(device)
    test_data = torch.tensor(test_data).transpose(1, 2).type(torch.FloatTensor).to(device)

    test_loader_0 = DataLoader(train_data, batch_size=min(args.batch_size, len(train_data)), shuffle=False,
                               num_workers=0, drop_last=False)
    test_loader_1 = DataLoader(test_data, batch_size=min(args.batch_size, len(test_data)), shuffle=False,
                               num_workers=0, drop_last=False)

    train_repr = model_test(model, test_loader_0)
    test_repr = model_test(model, test_loader_1)

    classifier = nn.Linear(64, args.num_class)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.1)

    train_repr = train_repr.tolist()
    train_labels = train_labels.tolist()
    test_repr = test_repr.tolist()
    test_labels = test_labels.tolist()

    train_pd = pd.DataFrame({"data": train_repr, "label": train_labels})
    data_train = dataset(list_data=train_pd)
    test_pd = pd.DataFrame({"data": test_repr, "label": test_labels})
    data_test = dataset(list_data=test_pd)

    train_loader = DataLoader(data_train, batch_size=64, shuffle=True,
                              num_workers=0,
                              drop_last=False)
    test_loader = DataLoader(data_test, batch_size=64, shuffle=True,
                             num_workers=0,
                             drop_last=False)

    classifier.to(device)
    classifier.train()
    for epoch in range(300):
        for batch in train_loader:
            x = batch[0]
            x = x.squeeze()
            y = batch[1]
            x = x.to(device)
            y = y.to(device)
            logits = classifier(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 200 or epoch == 250:
            scheduler.step()

    acc_all = accuracy(args, classifier, test_loader, device)

    return acc_all
