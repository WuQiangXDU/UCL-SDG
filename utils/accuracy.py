import torch
import copy


def accuracy(args, network, loader, device):
    network.eval()
    network = network.to(device)
    acc = []
    epoch_length_all = torch.zeros(args.num_dom + 1)
    correct_all = torch.zeros(args.num_dom + 1)

    with torch.no_grad():
        for ddd, (x, y) in enumerate(loader):
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            x = x.to(device)
            y = y.to(device)
            p = network(x)
            pred = p.argmax(dim=1)

            label = y
            test_labels_target = copy.deepcopy(y)

            # Harmonize labels from different domains
            for i in range(args.num_class):
                for id_domain in range(args.num_dom):
                    test_labels_target[test_labels_target == (i + id_domain * 10)] = test_labels_target[
                                                                                         test_labels_target == (
                                                                                                 i + id_domain * 10)] - id_domain * 10

            for i in range(args.num_class):
                data_temporary = []
                label_temporary = []
                for j in range(args.num_dom):
                    data_temporary.append(pred[label == (i + j * 10)])
                    label_temporary.append(label[label == (i + j * 10)])
                if i == 0:
                    domain_x = data_temporary
                    domain_y = label_temporary
                    continue
                for j in range(args.num_dom):
                    domain_x[j] = torch.cat((domain_x[j], data_temporary[j]), dim=0)
                    domain_y[j] = torch.cat((domain_y[j], label_temporary[j]), dim=0)

            for d in range(args.num_dom + 1):  # The result of a single target domain versus all target domains, the last one being all target domains
                if d > (args.num_dom - 1):
                    correct_d = torch.eq(pred, test_labels_target).float().sum().item()
                    epoch_length_all[d] = epoch_length_all[d] + label.size(0)
                    correct_all[d] = correct_all[d] + correct_d
                    break
                correct_d = torch.eq(domain_x[d], domain_y[d] - d * 10).float().sum().item()
                epoch_length_all[d] = epoch_length_all[d] + domain_x[d].size(0)
                correct_all[d] = correct_all[d] + correct_d

    for i in range(args.num_dom + 1):
        oa = correct_all[i] / epoch_length_all[i]
        acc.append(oa)

    return acc