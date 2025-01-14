import torch
import numpy as np
from models.losses import instance_contrastive_loss
from torch.utils.data import TensorDataset, Dataset, DataLoader
import models
from models.classifier import eval_classification

class UCL_SDG:
    '''The UCL_SDG model'''

    def __init__(
            self,
            args,
            lr,
            batch_size,
            after_iter_callback=None,
            after_epoch_callback=None
    ):

        super().__init__()
        self.device = torch.device("cuda")
        self.lr = lr
        self.batch_size = batch_size
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        self.n_epochs = 0
        self.n_iters = 0
        self.args = args
        self.model = getattr(models, args.model_name)()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)


    def fit(self, all_data_label, n_epochs=None, verbose=True):

        ''' Training the UCL_SDG model.

        Args:
            all_data_label (list): Training and test data, [0] is the source domain training data, [1] is the corresponding label; [2] is the target domain test data (include part of the source domain data not used for training), [3] is the corresponding label

        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''

        train_data = all_data_label[0]
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, num_workers=0,
                                  drop_last=False)

        if self.args.lr_scheduler == 'step':
            steps = [int(step) for step in self.args.steps.split(',')]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=self.args.gamma)
        elif self.args.lr_scheduler == 'exp':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.args.gamma)
        elif self.args.lr_scheduler == 'stepLR':
            steps = int(self.args.steps)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, steps, self.args.gamma)
        elif self.args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        self.model.to(self.device)

        loss_log = []

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            self.model.train()
            for batch in train_loader:
                x = batch[0]
                x = x.to(self.device)

                input_0 = x
                input_0 = input_0.cpu().detach().numpy()
                rand_num = np.random.uniform(1/self.args.hp, self.args.hp)
                arr_random = np.random.uniform(0, rand_num, size=(input_0.shape[0], input_0.shape[1], input_0.shape[2]))
                input_1 = np.multiply(input_0, arr_random)

                input = np.concatenate((input_0, input_1), axis=0)
                input = torch.from_numpy(input)
                input = input.to(self.device).float()
                input = input.transpose(1, 2)

                out = self.model(input)
                out = out.squeeze()
                out_0 = out.narrow(0, 0, int(out.size(0)/2))
                out_1 = out.narrow(0, int(out.size(0)/2), int(out.size(0)/2))

                loss = instance_contrastive_loss(out_0, out_1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cum_loss += loss.item()
                n_epoch_iters += 1

                self.n_iters += 1

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)

            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.n_epochs == 100 or self.n_epochs == 150:
                self.lr_scheduler.step()
                print('lr:', self.optimizer.state_dict()['param_groups'][0]['lr'])

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        acc = eval_classification(self.model, all_data_label, self.args, device=self.device)

        return loss_log, acc
